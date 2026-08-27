/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0A0D12",
          900: "#0F131A",
          800: "#141924",
          700: "#1C2230",
          600: "#2A3142",
        },
        paper: {
          100: "#EDE8DA",
          300: "#C9C3B3",
          500: "#8D8878",
        },
        amber: {
          400: "#F0B860",
          500: "#E8A33D",
          600: "#C9832A",
        },
        signal: {
          safe: "#3FA796",
          watch: "#E8A33D",
          risk: "#D2694F",
          critical: "#B34437",
        },
      },
      fontFamily: {
        display: ["Fraunces", "ui-serif", "Georgia", "serif"],
        body: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      backgroundImage: {
        "ledger-grid":
          "linear-gradient(rgba(237,232,218,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(237,232,218,0.035) 1px, transparent 1px)",
      },
      backgroundSize: {
        ledger: "28px 28px",
      },
      keyframes: {
        "print-in": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "print-in": "print-in 220ms ease-out",
      },
    },
  },
  plugins: [],
};
