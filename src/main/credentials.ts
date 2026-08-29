import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { safeStorage } from "electron";

export class CredentialStore {
  constructor(private readonly tokenPath: string) {}

  async hasToken(): Promise<boolean> {
    return (await this.readToken()) !== undefined;
  }

  async readToken(): Promise<string | undefined> {
    if (!safeStorage.isEncryptionAvailable()) return undefined;
    try {
      const encrypted = Buffer.from(await readFile(this.tokenPath, "utf8"), "base64");
      return safeStorage.decryptString(encrypted);
    } catch {
      return undefined;
    }
  }

  async saveToken(token: string): Promise<void> {
    const normalized = token.trim();
    if (!normalized.startsWith("hf_") || normalized.length < 20) {
      throw new Error("Enter a valid Hugging Face token beginning with hf_.");
    }
    if (!safeStorage.isEncryptionAvailable()) {
      throw new Error("macOS Keychain encryption is unavailable.");
    }
    await mkdir(dirname(this.tokenPath), { recursive: true });
    const encrypted = safeStorage.encryptString(normalized).toString("base64");
    await writeFile(this.tokenPath, encrypted, { encoding: "utf8", mode: 0o600 });
  }

  async deleteToken(): Promise<void> {
    await rm(this.tokenPath, { force: true });
  }
}
