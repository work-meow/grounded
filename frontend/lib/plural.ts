/**
 * A count with its noun in the right form.
 *
 * Russian picks between three: one файл, два файла, пять файлов — and the rule
 * is on the last digit, except in the teens, where it is not. Written once
 * because it was about to be written twice, in two different components, with
 * two chances of getting the teens wrong.
 */
export function plural(count: number, one: string, few: string, many: string): string {
  const tail = count % 10;
  const teens = count % 100;
  if (teens >= 11 && teens <= 14) return `${count} ${many}`;
  if (tail === 1) return `${count} ${one}`;
  if (tail >= 2 && tail <= 4) return `${count} ${few}`;
  return `${count} ${many}`;
}
