import "./globals.css";
import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

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
    <html lang="en">
      <body className={`${plusJakartaSans.className} antialiased selection:bg-accent selection:text-white bg-background text-textPrimary`}>
        {children}
      </body>
    </html>
  );
}


