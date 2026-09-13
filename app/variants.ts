export const VARIANTS = [
  "baseline",
  "improvement-1",
  "improvement-2",
  "improvement-3",
] as const;
export type Variant = (typeof VARIANTS)[number];
