/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,html}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif"
        ],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "monospace"]
      },
      colors: {
        ink: "#171717",
        panel: "#ffffff",
        line: "#d8dadd",
        field: "#f5f6f7",
        signal: "#0f766e",
        hazard: "#b42318",
        amberline: "#a15c07",
        violetless: "#2f6f9f"
      },
      boxShadow: {
        crisp: "0 1px 0 rgba(15, 23, 42, 0.08)"
      }
    }
  },
  plugins: []
};

