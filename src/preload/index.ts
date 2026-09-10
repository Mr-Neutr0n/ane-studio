import { contextBridge, ipcRenderer } from "electron";
import {
  type BootstrapData,
  isWorkerEvent,
  normalizeHuggingFaceToken,
  type StudioAPI,
  type StudioCommand,
  validateStudioRequest,
  type WorkerEvent,
} from "../shared/contracts";

const api: StudioAPI = {
  bootstrap: () => ipcRenderer.invoke("studio:bootstrap") as Promise<BootstrapData>,
  request: async <T>(command: StudioCommand, payload: Record<string, unknown> = {}) => {
    const validated = validateStudioRequest(command, payload);
    return (await ipcRenderer.invoke("studio:request", validated.command, validated.payload)) as T;
  },
  cancel: () => ipcRenderer.invoke("studio:cancel") as Promise<void>,
  saveToken: async (token: string) => {
    await ipcRenderer.invoke("studio:token:save", normalizeHuggingFaceToken(token));
  },
  deleteToken: () => ipcRenderer.invoke("studio:token:delete") as Promise<void>,
  onWorkerEvent: (callback: (event: WorkerEvent) => void) => {
    if (typeof callback !== "function")
      throw new Error("Worker event callback must be a function.");
    const listener = (_event: Electron.IpcRendererEvent, value: unknown): void => {
      if (isWorkerEvent(value)) callback(value);
    };
    ipcRenderer.on("studio:worker-event", listener);
    return () => ipcRenderer.removeListener("studio:worker-event", listener);
  },
};

contextBridge.exposeInMainWorld("studio", Object.freeze(api));
