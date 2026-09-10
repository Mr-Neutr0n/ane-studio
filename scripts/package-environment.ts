import { tmpdir } from "node:os";
import { join } from "node:path";

export function cleanEnvironment(home: string): Record<string, string> {
  return {
    HOME: home,
    PATH: "/usr/bin:/bin:/usr/sbin:/sbin",
    TMPDIR: tmpdir(),
    LANG: "en_US.UTF-8",
    ANE_STUDIO_HOME: join(home, "data"),
    HF_HOME: join(home, "huggingface"),
    HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
    HF_HUB_DISABLE_TELEMETRY: "1",
    DO_NOT_TRACK: "1",
    HF_TOKEN: "",
  };
}
