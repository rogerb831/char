import type { BatchParams, BatchResponse } from "@hypr/plugin-listener2";
import {
  commands as listener2Commands,
  events as listener2Events,
} from "@hypr/plugin-listener2";

export async function runBatchAwaitResponse(
  params: BatchParams,
): Promise<BatchResponse> {
  const sessionId = params.session_id;

  let unlisten: (() => void) | undefined;
  let settled = false;

  const cleanup = () => {
    if (unlisten) {
      unlisten();
      unlisten = undefined;
    }
  };

  return await new Promise<BatchResponse>((resolve, reject) => {
    listener2Events.batchEvent
      .listen(({ payload }) => {
        if (settled || payload.session_id !== sessionId) {
          return;
        }

        if (payload.type === "batchFailed") {
          settled = true;
          cleanup();
          reject(new Error(payload.error));
        }
      })
      .then((fn) => {
        unlisten = fn;

        listener2Commands
          .runBatch(params)
          .then((result) => {
            if (settled) {
              return;
            }

            if (result.status === "error") {
              settled = true;
              cleanup();
              reject(new Error(result.error));
              return;
            }

            settled = true;
            cleanup();
            resolve(result.data.response);
          })
          .catch((error) => {
            if (settled) {
              return;
            }

            settled = true;
            cleanup();
            reject(error instanceof Error ? error : new Error(String(error)));
          });
      })
      .catch((error) => {
        if (settled) {
          return;
        }

        settled = true;
        cleanup();
        reject(error instanceof Error ? error : new Error(String(error)));
      });
  });
}
