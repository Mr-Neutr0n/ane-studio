import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Use only local fixtures and static pages. No model downloads or external link visits.
test("loads without errors or external requests, with visible content and no overflow", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  const external: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("request", (request) => {
    if (!request.url().startsWith("http://127.0.0.1:4187/")) external.push(request.url());
  });
  const response = await page.goto("/");
  expect(response?.status()).toBe(200);
  expect(response?.headers()["content-security-policy"]).toContain("connect-src 'none'");
  await expect(page).toHaveTitle("ANE Studio | A local model lab for Apple Silicon");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Small models,meet your Mac.");
  await expect(page.getByRole("navigation", { name: "Main navigation" })).toBeVisible();
  for (const image of await page.locator("img").all()) {
    expect(
      await image.evaluate(
        (element) =>
          (element as HTMLImageElement).complete && (element as HTMLImageElement).naturalWidth > 0,
      ),
    ).toBe(true);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("homepage-full.png"), fullPage: true });
  await page.screenshot({ path: testInfo.outputPath("homepage-fold.png"), fullPage: false });
});

test("screenshot controls switch real app images and accessible state", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Chat locally", exact: true }).click();
  await expect(page.getByRole("button", { name: "Chat locally", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator("#app-screenshot")).toHaveAttribute("src", "/app-chat.webp");
  await expect(page.locator("#app-screenshot")).toHaveAttribute("alt", /local chat/);
  await expect(page.locator("#screenshot-link")).toHaveAttribute("href", "/app-chat.webp");
  await page.getByRole("button", { name: "Compare compute", exact: true }).click();
  await expect(page.locator("#app-screenshot")).toHaveAttribute("src", "/app-benchmark.webp");
  await expect(page.getByRole("button", { name: "Chat locally", exact: true })).toHaveAttribute(
    "aria-pressed",
    "false",
  );
});

test("source install action, copy success and clipboard denial are usable", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: async (text: string) => {
          document.documentElement.dataset.copied = text;
        },
      },
      configurable: true,
    });
  });
  await page.goto("/");
  const action = page.locator(".hero-actions .button-primary");
  const destination = await action.getAttribute("href");
  if (destination === "#install") {
    await action.click();
    await expect(page).toHaveURL(/#install$/);
    await expect(
      page.getByRole("heading", { name: "Put ANE Studio on your Mac." }),
    ).toBeInViewport();
  } else {
    expect(destination).toMatch(
      /^https:\/\/github\.com\/Mr-Neutr0n\/ane-studio\/releases\/download\/v\d+\.\d+\.\d+\/ANE-Studio-.*-arm64\.zip$/,
    );
  }
  await page.getByRole("button", { name: "Copy source install commands" }).click();
  await expect(page.getByRole("status")).toHaveText("Commands copied. Paste them into Terminal.");
  expect(await page.locator("html").getAttribute("data-copied")).toContain("bun run worker:setup");
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: async () => {
          throw new Error("Denied");
        },
      },
      configurable: true,
    });
  });
  await page.getByRole("button", { name: "Copy source install commands" }).click();
  await expect(page.getByRole("status")).toContainText("Clipboard unavailable");
  await expect(page.locator("#install-commands")).toBeVisible();
});

test("FAQ and skip navigation work with keyboard; reduced motion is respected", async ({
  page,
  browserName,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  // Safari's default Tab behavior skips links; Option+Tab includes all focusable controls.
  await page.keyboard.press(browserName === "webkit" ? "Alt+Tab" : "Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#main$/);
  const question = page
    .locator("summary")
    .filter({ hasText: "Does anything run in this website?" });
  await question.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText("There is no browser inference", { exact: false })).toBeVisible();
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior)).toBe(
    "auto",
  );
});

test("meets automated WCAG 2.2 AA checks in default and expanded states", async ({ page }) => {
  await page.goto("/");
  const closed = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(closed.violations).toEqual([]);
  await page.evaluate(() => {
    for (const details of document.querySelectorAll("details")) details.open = true;
  });
  const expanded = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(expanded.violations).toEqual([]);
});

test("essential content and source install remain available without JavaScript", async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false, baseURL });
  const page = await context.newPage();
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.locator("#install-commands")).toContainText("bun run dev");
  await page.locator("summary").filter({ hasText: "Is this a full-size chat assistant?" }).click();
  await expect(
    page.getByText("There is no stateful KV cache yet.", { exact: false }),
  ).toBeVisible();
  await context.close();
});
