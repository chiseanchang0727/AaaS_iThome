/**
 * Which account the app acts for. There is no login yet: the account is a
 * name kept in this browser (default "test_user") and sent as X-Account.
 */

export const DEFAULT_ACCOUNT = 'test_user'
const KEY = 'account'

export function getAccount(): string {
  try {
    return localStorage.getItem(KEY) || DEFAULT_ACCOUNT
  } catch {
    return DEFAULT_ACCOUNT
  }
}

export function accountHeaders(): Record<string, string> {
  return { 'X-Account': getAccount() }
}
