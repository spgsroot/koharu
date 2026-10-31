use std::collections::{BTreeMap, BTreeSet};

use koharu_scene::EntityId;

use crate::Stage;

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Pending,
    Running,
    Finished,
}

struct StageWork {
    stage: Stage,
    state: WorkState,
}

struct PageWork {
    page: EntityId,
    stages: Vec<StageWork>,
}

impl PageWork {
    fn started(&self) -> bool {
        self.stages
            .iter()
            .any(|work| work.state != WorkState::Pending)
    }

    fn finished(&self) -> bool {
        self.stages
            .iter()
            .all(|work| work.state == WorkState::Finished)
    }

    fn ready(&self, index: usize) -> bool {
        prerequisites(self.stages[index].stage)
            .iter()
            .all(|prerequisite| {
                self.stages
                    .iter()
                    .find(|work| work.stage == *prerequisite)
                    .is_none_or(|work| work.state == WorkState::Finished)
            })
    }
}

pub(crate) struct Scheduler {
    pages: Vec<PageWork>,
    page_index: BTreeMap<EntityId, usize>,
    page_window: usize,
    active_pages: usize,
    head: usize,
    translation_head: usize,
    total: usize,
}

impl Scheduler {
    pub(crate) fn new(pages: &[EntityId], stages: &[Stage]) -> Self {
        let pages = pages
            .iter()
            .map(|page| PageWork {
                page: *page,
                stages: stages
                    .iter()
                    .map(|stage| StageWork {
                        stage: *stage,
                        state: WorkState::Pending,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let total = pages.len().saturating_mul(stages.len());
        Self {
            page_index: pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.page, index))
                .collect(),
            pages,
            page_window: stages.len().max(1),
            active_pages: 0,
            head: 0,
            translation_head: 0,
            total,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn start_next(
        &mut self,
        busy_stages: &BTreeSet<Stage>,
    ) -> Option<(EntityId, Stage)> {
        for page_index in self.head..self.pages.len() {
            let started = self.pages[page_index].started();
            if !started && self.active_pages >= self.page_window {
                break;
            }
            let stage_index =
                self.pages[page_index]
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(index, work)| {
                        (work.state == WorkState::Pending
                            && !busy_stages.contains(&work.stage)
                            && (work.stage != Stage::Translation
                                || page_index == self.translation_head)
                            && self.pages[page_index].ready(index))
                        .then_some(index)
                    });
            let Some(stage_index) = stage_index else {
                continue;
            };
            if !started {
                self.active_pages += 1;
            }
            let page = &mut self.pages[page_index];
            let work = &mut page.stages[stage_index];
            work.state = WorkState::Running;
            return Some((page.page, work.stage));
        }
        None
    }

    pub(crate) fn complete_stage(&mut self, page: EntityId, stage: Stage) -> bool {
        let Some(&page_index) = self.page_index.get(&page) else {
            return false;
        };
        let page = &mut self.pages[page_index];
        let was_finished = page.finished();
        if let Some(work) = page.stages.iter_mut().find(|work| work.stage == stage) {
            work.state = WorkState::Finished;
            if stage == Stage::Translation && page_index == self.translation_head {
                self.translation_head += 1;
            }
        }
        let page_finished = !was_finished && page.finished();
        if page_finished {
            self.active_pages = self.active_pages.saturating_sub(1);
            while self.head < self.pages.len() && self.pages[self.head].finished() {
                self.head += 1;
            }
        }
        page_finished
    }

    /// Returns a running stage to the queue, for example after its output
    /// was superseded by a concurrent edit.
    pub(crate) fn retry(&mut self, page: EntityId, stage: Stage) {
        let Some(&page_index) = self.page_index.get(&page) else {
            return;
        };
        if let Some(work) = self.pages[page_index]
            .stages
            .iter_mut()
            .find(|work| work.stage == stage && work.state == WorkState::Running)
        {
            work.state = WorkState::Pending;
        }
    }
}

const fn prerequisites(stage: Stage) -> &'static [Stage] {
    match stage {
        Stage::Detection => &[],
        Stage::CensorDetection => &[Stage::Detection],
        // OCR and cleanup observe the page's asset collection. Commit its
        // censor-mask write before either reader captures that collection.
        Stage::Ocr | Stage::Inpainting => &[Stage::Detection, Stage::CensorDetection],
        Stage::Translation => &[Stage::Ocr],
        Stage::Decensor => &[Stage::CensorDetection, Stage::Inpainting],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(count: usize) -> Vec<EntityId> {
        (0..count).map(|_| EntityId::new()).collect()
    }

    #[test]
    fn a_superseded_translation_reruns_before_later_pages_translate() {
        let pages = pages(2);
        let stages = [Stage::Ocr, Stage::Translation];
        let mut scheduler = Scheduler::new(&pages, &stages);
        let busy = BTreeSet::new();
        assert_eq!(scheduler.start_next(&busy), Some((pages[0], Stage::Ocr)));
        scheduler.complete_stage(pages[0], Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Translation))
        );
        assert_eq!(scheduler.start_next(&busy), Some((pages[1], Stage::Ocr)));
        scheduler.complete_stage(pages[1], Stage::Ocr);
        scheduler.retry(pages[0], Stage::Translation);
        // The retried page keeps its place at the translation head, so prior-page
        // memory still reaches the next page.
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Translation))
        );
        assert!(scheduler.start_next(&busy).is_none());
        scheduler.complete_stage(pages[0], Stage::Translation);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[1], Stage::Translation))
        );
    }

    #[test]
    fn page_asset_readers_wait_for_censor_mask_but_other_pages_keep_running() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL);
        let mut busy = BTreeSet::new();

        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Detection))
        );
        scheduler.complete_stage(pages[0], Stage::Detection);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::CensorDetection))
        );
        busy.insert(Stage::CensorDetection);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[1], Stage::Detection))
        );
        busy.insert(Stage::Detection);
        assert!(scheduler.start_next(&busy).is_none());

        scheduler.complete_stage(pages[0], Stage::CensorDetection);
        busy.remove(&Stage::CensorDetection);
        assert_eq!(scheduler.start_next(&busy), Some((pages[0], Stage::Ocr)));
        busy.insert(Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Inpainting))
        );
        busy.insert(Stage::Inpainting);
        assert!(scheduler.start_next(&busy).is_none());

        scheduler.complete_stage(pages[0], Stage::Ocr);
        busy.remove(&Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Translation))
        );
        busy.insert(Stage::Translation);
        scheduler.complete_stage(pages[0], Stage::Inpainting);
        busy.remove(&Stage::Inpainting);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Decensor))
        );
    }

    #[test]
    fn sliding_window_backpressures_fast_upstream_models() {
        let pages = pages(4);
        let stages = [
            Stage::Detection,
            Stage::Ocr,
            Stage::CensorDetection,
            Stage::Inpainting,
        ];
        let mut scheduler = Scheduler::new(&pages, &stages);
        let mut busy = BTreeSet::new();

        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Detection))
        );
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::CensorDetection))
        );
        assert!(!scheduler.complete_stage(pages[0], Stage::CensorDetection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        busy.insert(inpainting.1);

        for page in &pages[1..4] {
            assert_eq!(scheduler.start_next(&busy), Some((*page, Stage::Detection)));
            assert!(!scheduler.complete_stage(*page, Stage::Detection));
            assert_eq!(
                scheduler.start_next(&busy),
                Some((*page, Stage::CensorDetection))
            );
            assert!(!scheduler.complete_stage(*page, Stage::CensorDetection));
        }
        assert!(scheduler.start_next(&busy).is_none());

        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        assert!(scheduler.complete_stage(pages[0], Stage::Inpainting));
        busy.clear();
        assert_eq!(scheduler.start_next(&busy), Some((pages[1], Stage::Ocr)));
    }

    #[test]
    fn a_later_ready_page_cannot_translate_before_earlier_memory_is_committed() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &[Stage::Ocr, Stage::Translation]);
        scheduler.complete_stage(pages[1], Stage::Ocr);
        let busy = BTreeSet::from([Stage::Ocr]);
        assert!(scheduler.start_next(&busy).is_none());
        scheduler.complete_stage(pages[0], Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Translation))
        );
        assert!(scheduler.start_next(&busy).is_none());
        scheduler.complete_stage(pages[0], Stage::Translation);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[1], Stage::Translation))
        );
    }
}
