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
        ink: "#4260ff",
        panel: "#03040b",
        line: "#22306f",
        field: "#080b18",
        signal: "#4260ff",
        hazard: "#ff4d5e",
        amberline: "#d7ddff",
        violetless: "#8fa1ff"
      },
      boxShadow: {
        crisp: "0 12px 34px rgba(0, 0, 0, 0.28)"
      }
    }
  },
  plugins: []
};
