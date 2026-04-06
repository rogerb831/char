import { commands as fsSyncCommands } from "@hypr/plugin-fs-sync";
import type { BatchParams } from "@hypr/plugin-listener2";

import { providerRowId } from "~/settings/ai/shared";
import { id } from "~/shared/utils";
import type { Store as MainStore } from "~/store/tinybase/store/main";
import type { Store as SettingsStore } from "~/store/tinybase/store/settings";
import { transformBatchResponse } from "~/store/zustand/listener/batch";
import { runBatchAwaitResponse } from "~/stt/run-batch-await";
import type { RuntimeSpeakerHint, WordLike } from "~/stt/segment";
import type { SpeakerHintWithId, WordWithId } from "~/stt/types";
import { extractKeywordsFromMarkdown } from "~/stt/useKeywords";
import { updateTranscriptHints, updateTranscriptWords } from "~/stt/utils";

function getWatsonxSttConnection(
  settingsStore: SettingsStore,
): { model: string; baseUrl: string; apiKey: string } | null {
  const provider = settingsStore.getValue("current_stt_provider");
  if (provider !== "watsonx") {
    return null;
  }

  const model = settingsStore.getValue("current_stt_model");
  if (typeof model !== "string" || !model) {
    return null;
  }

  const rowId = providerRowId("stt", "watsonx");
  const baseUrl = String(
    settingsStore.getCell("ai_providers", rowId, "base_url") ?? "",
  ).trim();
  const apiKey = String(
    settingsStore.getCell("ai_providers", rowId, "api_key") ?? "",
  ).trim();

  if (!baseUrl || !apiKey) {
    return null;
  }

  return { model, baseUrl, apiKey };
}

function getSpokenLanguageCodes(settingsStore: SettingsStore): string[] {
  const raw = settingsStore.getValue("spoken_languages");
  if (typeof raw !== "string" || !raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((code): code is string => typeof code === "string");
  } catch {
    return [];
  }
}

function keywordsForSession(rawMd: string): string[] {
  const { keywords, keyphrases } = extractKeywordsFromMarkdown(rawMd);
  return Array.from(new Set([...keywords, ...keyphrases])).filter(
    (k) => k.length >= 2,
  );
}

function replaceTranscriptWithBatchWords(
  store: MainStore,
  transcriptId: string,
  provider: string,
  words: WordLike[],
  hints: RuntimeSpeakerHint[],
): void {
  const newWords: WordWithId[] = [];
  const newWordIds: string[] = [];

  words.forEach((word) => {
    const wordId = id();
    newWords.push({
      id: wordId,
      text: word.text,
      start_ms: word.start_ms,
      end_ms: word.end_ms,
      channel: word.channel,
    });
    newWordIds.push(wordId);
  });

  const newHints: SpeakerHintWithId[] = [];

  hints.forEach((hint) => {
    if (hint.data.type !== "provider_speaker_index") {
      return;
    }

    const wordId = newWordIds[hint.wordIndex];
    const word = words[hint.wordIndex];

    if (!wordId || !word) {
      return;
    }

    newHints.push({
      id: id(),
      word_id: wordId,
      type: "provider_speaker_index",
      value: JSON.stringify({
        provider: hint.data.provider ?? provider,
        channel: hint.data.channel ?? word.channel,
        speaker_index: hint.data.speaker_index,
      }),
    });
  });

  updateTranscriptWords(store, transcriptId, newWords);
  updateTranscriptHints(store, transcriptId, newHints);
}

function clearTranscriptWords(store: MainStore, transcriptId: string): void {
  updateTranscriptWords(store, transcriptId, []);
  updateTranscriptHints(store, transcriptId, []);
}

export async function runWatsonBatchSttBeforeEnhance(
  sessionId: string,
  store: MainStore,
  settingsStore: SettingsStore,
): Promise<void> {
  const conn = getWatsonxSttConnection(settingsStore);
  if (!conn) {
    return;
  }

  const audioResult = await fsSyncCommands.audioPath(sessionId);
  if (audioResult.status !== "ok" || !audioResult.data) {
    return;
  }

  const filePath = audioResult.data;
  const rawMd = store.getCell("sessions", sessionId, "raw_md");
  const keywords = typeof rawMd === "string" ? keywordsForSession(rawMd) : [];

  const transcriptIds: string[] = [];
  store.forEachRow("transcripts", (transcriptId, _forEachCell) => {
    const sid = store.getCell("transcripts", transcriptId, "session_id");
    if (sid === sessionId) {
      transcriptIds.push(transcriptId);
    }
  });

  if (transcriptIds.length === 0) {
    return;
  }

  const primaryId = transcriptIds.reduce((best, rowId) => {
    const started =
      (store.getCell("transcripts", rowId, "started_at") as
        | number
        | undefined) ?? 0;
    const bestStarted =
      (store.getCell("transcripts", best, "started_at") as
        | number
        | undefined) ?? 0;
    return started <= bestStarted ? rowId : best;
  });

  const params: BatchParams = {
    session_id: sessionId,
    provider: "watsonx",
    file_path: filePath,
    model: conn.model,
    base_url: conn.baseUrl,
    api_key: conn.apiKey,
    keywords,
    languages: getSpokenLanguageCodes(settingsStore),
  };

  const response = await runBatchAwaitResponse(params);
  const [words, hints] = transformBatchResponse(response);

  if (words.length === 0) {
    return;
  }

  replaceTranscriptWithBatchWords(store, primaryId, "watsonx", words, hints);

  for (const tid of transcriptIds) {
    if (tid !== primaryId) {
      clearTranscriptWords(store, tid);
    }
  }
}
