import { describe, expect, it } from "vitest";
import { citationSchema } from "./api";

describe("response validation", () => {
  it("accepts a citation with only its required fields", () => {
    const parsed = citationSchema.safeParse({
      index: 1,
      chunk_id: "chk_1",
      document_id: "doc_1",
      document_title: "Handbook",
      snippet: "Up to 10 days may be carried over.",
    });
    expect(parsed.success).toBe(true);
  });

  it("rejects a citation missing its document, rather than rendering a blank source", () => {
    const parsed = citationSchema.safeParse({ index: 1, chunk_id: "chk_1", snippet: "x" });
    expect(parsed.success).toBe(false);
  });

  it("defaults section_path so a component never indexes into undefined", () => {
    const parsed = citationSchema.parse({
      index: 2,
      chunk_id: "c",
      document_id: "d",
      document_title: "t",
      snippet: "s",
    });
    expect(parsed.section_path).toEqual([]);
  });
});
