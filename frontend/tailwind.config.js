/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,html}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Courier New",
          "Courier",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace"
        ],
        mono: ["Courier New", "Courier", "SFMono-Regular", "Menlo", "monospace"]
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
        crisp: "0 0 0 1px rgba(66, 96, 255, 0.22), 0 0 28px rgba(66, 96, 255, 0.1)"
      }
    }
  },
  plugins: []
};
