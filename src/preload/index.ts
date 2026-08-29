import { contextBridge, ipcRenderer } from "electron";
import type { BootstrapData, StudioAPI, WorkerEvent } from "../shared/contracts";

const api: StudioAPI = {
  bootstrap: () => ipcRenderer.invoke("studio:bootstrap") as Promise<BootstrapData>,
  request: <T>(command: string, payload: Record<string, unknown> = {}) =>
    ipcRenderer.invoke("studio:request", command, payload) as Promise<T>,
  cancel: () => ipcRenderer.invoke("studio:cancel") as Promise<void>,
  saveToken: (token: string) => ipcRenderer.invoke("studio:token:save", token) as Promise<void>,
  deleteToken: () => ipcRenderer.invoke("studio:token:delete") as Promise<void>,
  onWorkerEvent: (callback: (event: WorkerEvent) => void) => {
    ipcRenderer.removeAllListeners("studio:worker-event");
    ipcRenderer.on("studio:worker-event", (_event, value: WorkerEvent) => callback(value));
  },
};

contextBridge.exposeInMainWorld("studio", api);
