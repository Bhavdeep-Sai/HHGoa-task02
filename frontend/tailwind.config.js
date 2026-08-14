/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'rgba(var(--background), <alpha-value>)',
        surface: 'rgba(var(--surface), <alpha-value>)',
        surfaceHover: 'rgba(var(--surface-hover), <alpha-value>)',
        border: 'rgba(var(--border), <alpha-value>)',
        textPrimary: 'rgba(var(--text-primary), <alpha-value>)',
        textSecondary: 'rgba(var(--text-secondary), <alpha-value>)',
        textTertiary: 'rgba(var(--text-tertiary), <alpha-value>)',
        bgDarker: 'rgba(var(--bg-darker), <alpha-value>)',
        accent: 'rgba(var(--accent), <alpha-value>)',
        accentGlow: 'var(--accent-glow)',
        primary: '#3b82f6',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
      },
    },
  },
  plugins: [],
}
