import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "IndicVoiceRAG — Adaptive Multilingual Voice RAG",
  description: "Ask. Retrieve. Verify. In any Indian language. Built for HackerHouse Goa 2026 Task 2.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased selection:bg-accent selection:text-white bg-background text-gray-100">
        {children}
      </body>
    </html>
  );
}
