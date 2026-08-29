import type { StudioAPI } from "../shared/contracts";

declare global {
  interface Window {
    studio: StudioAPI;
  }
}
