import { defineConfig, devices } from "@playwright/test";

/**
 * E2E configuration.
 *
 * These tests hit a real model, so they are slow and cost money. They run on a
 * schedule and before a release, not on every push — a suite people cancel
 * because it takes twenty minutes is a suite that stops being read.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // shared backend state; parallel runs interfere
  forbidOnly: !!process.env.CI,
  // One retry in CI, none locally. Retrying locally hides a flake from the
  // person best placed to fix it.
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 90_000,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : [["list"]],

  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    // Trace and screenshot only on a failure that survived its retry. Tracing
    // everything produces gigabytes of artifacts nobody opens.
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },

  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],

  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: "npm run build && npm run start",
        url: "http://localhost:3000",
        reuseExistingServer: !process.env.CI,
        timeout: 180_000,
      },
});
