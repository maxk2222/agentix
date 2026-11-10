/**
 * DOM utility functions
 */

/**
 * Create a DOM element with attributes
 */
export function createElement(
  tag: string,
  attrs: Record<string, string> = {},
  html: string = ''
): HTMLElement {
  const element = document.createElement(tag);
  
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === 'class') {
      element.className = value;
    } else if (key === 'id') {
      element.id = value;
    } else if (key.startsWith('aria-')) {
      element.setAttribute(key, value);
    } else if (key.startsWith('data-')) {
      element.setAttribute(key, value);
    } else {
      element.setAttribute(key, value);
    }
  });

  if (html) {
    element.innerHTML = html;
  }

  return element;
}

/**
 * Add event listener with cleanup
 */
export function addEvent(
  element: HTMLElement,
  event: string,
  handler: EventListener,
  options?: AddEventListenerOptions
): () => void {
  element.addEventListener(event, handler, options);
  return () => element.removeEventListener(event, handler);
}

/**
 * Get CSS variable value
 */
export function getCSSVariable(varName: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

/**
 * Scroll element to bottom
 */
export function scrollToBottom(element: HTMLElement): void {
  element.scrollTop = element.scrollHeight;
}

/**
 * Check if element is in viewport
 */
export function isInViewport(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

/**
 * Animate element fade in
 */
export function fadeIn(element: HTMLElement, duration: number = 300): Promise<void> {
  return new Promise((resolve) => {
    element.style.opacity = '0';
    element.style.transition = `opacity ${duration}ms ease-in`;
    
    // Trigger reflow
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    element.offsetHeight;
    
    element.style.opacity = '1';
    
    setTimeout(() => {
      element.style.transition = '';
      resolve();
    }, duration);
  });
}

/**
 * Animate element fade out
 */
export function fadeOut(element: HTMLElement, duration: number = 300): Promise<void> {
  return new Promise((resolve) => {
    element.style.opacity = '1';
    element.style.transition = `opacity ${duration}ms ease-out`;
    
    // Trigger reflow
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    element.offsetHeight;
    
    element.style.opacity = '0';
    
    setTimeout(() => {
      element.style.transition = '';
      resolve();
    }, duration);
  });
}
