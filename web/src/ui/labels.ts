export function humanizeLabel(value: string): string {
  const normalized = String(value || '')
    .trim()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')

  if (!normalized) return ''
  if (/^[A-Z0-9 ]+$/.test(normalized)) return normalized

  return normalized
    .split(' ')
    .map((word) => {
      if (/^[A-Z0-9]+$/.test(word)) return word
      if (word.toLowerCase() === 'id') return 'ID'
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    })
    .join(' ')
}
