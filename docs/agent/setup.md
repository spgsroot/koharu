---
title: Set Up Koharu Agent
description: Sign in with Codex, Grok or Kimi Code and choose the model and reasoning level used by Koharu Agent.
---

# Set Up Koharu Agent

Koharu Agent is a project-aware client built into the editor. Choose Codex, which uses a ChatGPT subscription, Grok, which uses an xAI account, or Kimi Code, which uses a Kimi Code subscription. All three use device-code authentication and do not ask for an API key. Each provider keeps its own sign-in.

## Sign in

1. Open a project and select the **Agent** panel on the right.
2. Choose **Codex**, **Grok** or **Kimi Code**, then sign in.
3. Koharu opens the device authorization page in your browser.
4. Enter the displayed device code and approve the sign-in.
5. Return to Koharu and wait for the account and model list to load.

Only one sign-in attempt can run at a time. Cancel it before starting another.

## Configure the agent

Choose a model and reasoning level for the selected provider. Higher reasoning levels can help with long, ambiguous project edits but take more time. Switching providers clears the current conversation but does not delete the other provider's sign-in. Koharu saves the agent configuration separately from translation-provider settings.

The Agent uses the open project's semantic state as its starting context. It does not automatically upload every page image. A page is rendered and attached only when the agent invokes its visual page-inspection tool.

## Account boundary

Agent authentication is separate from credentials under **Settings -> Providers**. Signing into Grok makes its language models available in the model picker; it does not change the selected translator until you choose one. Kimi Code is available to Agent only; it is not a translation provider. An API key does not sign in Koharu Agent.

Use **Sign out** to cancel active work, clear the current agent session, and remove only the selected provider's authorization.

Continue with [Work with projects](/agent/work-with-projects/).
