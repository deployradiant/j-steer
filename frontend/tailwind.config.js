import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx,html}"],
  theme: {
    extend: {
      fontWeight: {
        normal: 400,
        bold: 600
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "monospace"]
      },
      colors: {
        primary: {
          95: "var(--primary-95)",
          90: "var(--primary-90)",
          80: "var(--primary-80)",
          70: "var(--primary-70)",
          60: "var(--primary-60)",
          50: "var(--primary-50)",
          40: "var(--primary-40)",
          30: "var(--primary-30)",
          20: "var(--primary-20)",
          10: "var(--primary-10)"
        },
        buttercup: {
          100: "var(--buttercup-100)",
          90: "var(--buttercup-90)",
          80: "var(--buttercup-80)",
          70: "var(--buttercup-70)",
          60: "var(--buttercup-60)",
          50: "var(--buttercup-50)",
          40: "var(--buttercup-40)",
          30: "var(--buttercup-30)",
          20: "var(--buttercup-20)",
          10: "var(--buttercup-10)"
        },
        flame: {
          100: "var(--flame-100)",
          90: "var(--flame-90)",
          80: "var(--flame-80)",
          70: "var(--flame-70)",
          60: "var(--flame-60)",
          50: "var(--flame-50)",
          40: "var(--flame-40)",
          30: "var(--flame-30)",
          20: "var(--flame-20)",
          10: "var(--flame-10)"
        },
        slime: {
          100: "var(--slime-100)",
          90: "var(--slime-90)",
          80: "var(--slime-80)",
          70: "var(--slime-70)",
          60: "var(--slime-60)",
          50: "var(--slime-50)",
          40: "var(--slime-40)",
          30: "var(--slime-30)",
          20: "var(--slime-20)",
          10: "var(--slime-10)"
        },
        "dark-rose": {
          100: "var(--dark-rose-100)",
          90: "var(--dark-rose-90)",
          80: "var(--dark-rose-80)",
          70: "var(--dark-rose-70)",
          60: "var(--dark-rose-60)",
          50: "var(--dark-rose-50)",
          40: "var(--dark-rose-40)",
          30: "var(--dark-rose-30)",
          20: "var(--dark-rose-20)",
          10: "var(--dark-rose-10)"
        },
        grey: {
          100: "var(--grey-100)",
          95: "var(--grey-95)",
          90: "var(--grey-90)",
          85: "var(--grey-85)",
          80: "var(--grey-80)",
          70: "var(--grey-70)",
          60: "var(--grey-60)",
          50: "var(--grey-50)",
          40: "var(--grey-40)",
          30: "var(--grey-30)",
          20: "var(--grey-20)",
          10: "var(--grey-10)",
          0: "var(--grey-0)"
        },
        ink: "var(--primary-50)",
        panel: "var(--grey-0)",
        line: "var(--primary-20)",
        field: "#050713",
        signal: "var(--primary-50)",
        hazard: "var(--flame-60)",
        amberline: "var(--buttercup-60)",
        violetless: "var(--primary-70)"
      },
      boxShadow: {
        crisp: "0 12px 34px rgba(0, 0, 0, 0.28)"
      }
    }
  },
  plugins: [typography]
};
