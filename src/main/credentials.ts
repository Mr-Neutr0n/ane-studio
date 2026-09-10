import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { chmod, lstat, mkdir, open, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { safeStorage } from "electron";
import { normalizeHuggingFaceToken } from "../shared/contracts";

function encryptionAvailable(): boolean {
  return (
    safeStorage.isEncryptionAvailable() &&
    // Electron's Linux basic_text backend is not secret storage.
    (process.platform !== "linux" || safeStorage.getSelectedStorageBackend() !== "basic_text")
  );
}

export class CredentialStore {
  constructor(private readonly tokenPath: string) {}

  async hasToken(): Promise<boolean> {
    return (await this.readToken()) !== undefined;
  }

  async readToken(): Promise<string | undefined> {
    if (!encryptionAvailable()) return undefined;
    try {
      const directory = await lstat(dirname(this.tokenPath));
      if (!directory.isDirectory() || directory.isSymbolicLink()) return undefined;
      const file = await open(
        this.tokenPath,
        constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
      );
      try {
        const stat = await file.stat();
        if (!stat.isFile() || stat.size === 0 || stat.size > 16_384) return undefined;
        const encoded = await file.readFile("utf8");
        const encrypted = Buffer.from(encoded, "base64");
        if (encrypted.toString("base64") !== encoded) return undefined;
        return normalizeHuggingFaceToken(safeStorage.decryptString(encrypted));
      } finally {
        await file.close();
      }
    } catch {
      // Missing, damaged, or inaccessible credentials never fall back to plaintext.
      return undefined;
    }
  }

  async saveToken(token: unknown): Promise<void> {
    const normalized = normalizeHuggingFaceToken(token);
    if (!encryptionAvailable()) {
      throw new Error(
        "Secure credential encryption is unavailable. Unlock macOS Keychain and try again.",
      );
    }
    const directory = dirname(this.tokenPath);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const stat = await lstat(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      throw new Error("The credential directory must be a local directory, not a symbolic link.");
    }
    await chmod(directory, 0o700);
    const encrypted = safeStorage.encryptString(normalized).toString("base64");
    const temporary = `${this.tokenPath}.${randomUUID()}.tmp`;
    try {
      // A rename prevents partial reads and replaces old, overly broad file permissions.
      await writeFile(temporary, encrypted, { encoding: "utf8", mode: 0o600, flag: "wx" });
      await rename(temporary, this.tokenPath);
    } finally {
      await rm(temporary, { force: true });
    }
  }

  async deleteToken(): Promise<void> {
    await rm(this.tokenPath, { force: true });
  }
}
