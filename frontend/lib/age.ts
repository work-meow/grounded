/**
 * How long ago something was, in words.
 *
 * Shown next to a citation because an answer taken from a document nobody has
 * touched in two years is a right answer to an old question — and that belongs
 * beside the source, not inside the sentence, where the model would have to be
 * told to write it and would sometimes forget.
 *
 * Only says anything once it is worth saying: a document edited this week is
 * simply current, and "обновлён 3 дня назад" on every chip is noise that
 * teaches the reader to stop looking.
 */
const MONTH = 30 * 24 * 3600;

/** Below this, nothing is said at all. */
export const STALE_AFTER = 6 * MONTH;

export function staleness(modifiedAt?: number | null): string | null {
  if (!modifiedAt) return null;
  const seconds = Date.now() / 1000 - modifiedAt;
  if (seconds < STALE_AFTER) return null;
  const years = Math.floor(seconds / (12 * MONTH));
  if (years >= 1) return years === 1 ? "обновлён год назад" : `обновлён ${years} г. назад`;
  return `обновлён ${Math.floor(seconds / MONTH)} мес. назад`;
}
