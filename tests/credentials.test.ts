// @vitest-environment node
import {
  chmod,
  lstat,
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CredentialStore } from "../src/main/credentials";

const secure = vi.hoisted(() => ({
  available: vi.fn(),
  backend: vi.fn(),
  encrypt: vi.fn(),
  decrypt: vi.fn(),
  values: new Map<string, string>(),
}));
vi.mock("electron", () => ({
  safeStorage: {
    isEncryptionAvailable: secure.available,
    getSelectedStorageBackend: secure.backend,
    encryptString: secure.encrypt,
    decryptString: secure.decrypt,
  },
}));

const token = `hf_${"a".repeat(34)}`;
let directory: string;
let tokenPath: string;
let store: CredentialStore;
beforeEach(async () => {
  vi.clearAllMocks();
  secure.values.clear();
  secure.available.mockReturnValue(true);
  secure.backend.mockReturnValue("gnome_libsecret");
  secure.encrypt.mockImplementation((value: string) => {
    const key = `os-encrypted-value-${secure.values.size}`;
    secure.values.set(key, value);
    return Buffer.from(key);
  });
  secure.decrypt.mockImplementation((encrypted: Buffer) => {
    const value = secure.values.get(encrypted.toString());
    if (!value) throw new Error("Cannot decrypt.");
    return value;
  });
  directory = await mkdtemp(join(tmpdir(), "ane-credentials-test-"));
  tokenPath = join(directory, "credentials", "hf-token");
  store = new CredentialStore(tokenPath);
});
afterEach(async () => {
  await rm(directory, { force: true, recursive: true });
});

describe("local encrypted credential storage", () => {
  it("writes only safeStorage ciphertext with private file and directory permissions", async () => {
    await store.saveToken(` ${token} `);
    expect(secure.encrypt).toHaveBeenCalledExactlyOnceWith(token);
    const disk = await readFile(tokenPath, "utf8");
    expect(disk).not.toContain(token);
    expect(Buffer.from(disk, "base64").toString()).toBe("os-encrypted-value-0");
    expect((await lstat(tokenPath)).mode & 0o777).toBe(0o600);
    expect((await lstat(dirname(tokenPath))).mode & 0o777).toBe(0o700);
    expect(await store.readToken()).toBe(token);
    expect(await store.hasToken()).toBe(true);
    expect(await readdir(dirname(tokenPath))).toEqual(["hf-token"]);
  });

  it("repairs permissive files and directories when replacing an old token", async () => {
    await store.saveToken(token);
    await chmod(tokenPath, 0o644);
    await chmod(dirname(tokenPath), 0o755);
    const replacement = `hf_${"b".repeat(34)}`;
    await store.saveToken(replacement);
    expect(await store.readToken()).toBe(replacement);
    expect((await lstat(tokenPath)).mode & 0o777).toBe(0o600);
    expect((await lstat(dirname(tokenPath))).mode & 0o777).toBe(0o700);
  });

  it("never stores plaintext if Keychain encryption is unavailable", async () => {
    secure.available.mockReturnValue(false);
    await expect(store.saveToken(token)).rejects.toThrow("encryption is unavailable");
    expect(secure.encrypt).not.toHaveBeenCalled();
    expect(await store.readToken()).toBeUndefined();
    await expect(lstat(tokenPath)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("does not erase the previous token if encryption fails", async () => {
    await store.saveToken(token);
    const original = await readFile(tokenPath);
    secure.encrypt.mockImplementationOnce(() => {
      throw new Error("Keychain locked");
    });
    await expect(store.saveToken(`hf_${"b".repeat(34)}`)).rejects.toThrow();
    expect(await readFile(tokenPath)).toEqual(original);
    expect(await store.readToken()).toBe(token);
    expect(await readdir(dirname(tokenPath))).toEqual(["hf-token"]);
  });

  it("cleans a temporary file when an atomic replacement fails", async () => {
    await mkdir(tokenPath, { recursive: true });
    await expect(store.saveToken(token)).rejects.toThrow();
    expect(await readdir(dirname(tokenPath))).toEqual(["hf-token"]);
    expect((await lstat(tokenPath)).isDirectory()).toBe(true);
  });

  it("rejects malformed, oversized, and decrypted non-token contents", async () => {
    await mkdir(dirname(tokenPath), { recursive: true });
    for (const content of ["", "not-base64!", "x".repeat(16_385)]) {
      await writeFile(tokenPath, content);
      expect(await store.readToken()).toBeUndefined();
    }
    expect(secure.decrypt).not.toHaveBeenCalled();
    await writeFile(tokenPath, Buffer.from("unrecognized-cipher").toString("base64"));
    expect(await store.hasToken()).toBe(false);
    secure.decrypt.mockReturnValueOnce("not-a-token");
    expect(await store.readToken()).toBeUndefined();
  });

  it("does not follow token symlinks and does not overwrite their targets", async () => {
    const target = join(directory, "outside");
    await writeFile(target, "untouched");
    await mkdir(dirname(tokenPath));
    await symlink(target, tokenPath);
    expect(await store.readToken()).toBeUndefined();
    await store.saveToken(token);
    expect((await lstat(tokenPath)).isSymbolicLink()).toBe(false);
    expect(await readFile(target, "utf8")).toBe("untouched");
    expect(await store.readToken()).toBe(token);
  });

  it("rejects a symlinked credential directory", async () => {
    const target = join(directory, "outside");
    await mkdir(target);
    await symlink(target, dirname(tokenPath));
    await expect(store.saveToken(token)).rejects.toThrow("symbolic link");
    expect(await store.readToken()).toBeUndefined();
    expect(await readdir(target)).toEqual([]);
  });

  it("leaves a complete encrypted value after concurrent writes", async () => {
    const tokens = ["a", "b", "c", "d"].map((character) => `hf_${character.repeat(34)}`);
    await Promise.all(tokens.map((value) => store.saveToken(value)));
    expect(tokens).toContain(await store.readToken());
    expect(await readdir(dirname(tokenPath))).toEqual(["hf-token"]);
  });

  it("deletes idempotently, including when Keychain is locked", async () => {
    await store.saveToken(token);
    secure.available.mockReturnValue(false);
    await store.deleteToken();
    await store.deleteToken();
    expect(await store.hasToken()).toBe(false);
    await expect(lstat(tokenPath)).rejects.toMatchObject({ code: "ENOENT" });
  });
});
