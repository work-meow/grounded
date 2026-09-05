import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin", "cyrillic"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin", "cyrillic"],
});

export const metadata: Metadata = {
  title: "База знаний",
  description: "Личная RAG-система по вашим документам",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="ru"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      {/* The app is a shell, not a document: exactly the viewport tall, and it
          does not scroll. Every scrollable region inside owns its own scroll
          bar. With `min-h-full` here instead, a long chat list grew the body
          and took the header, the navigation and the composer with it. */}
      <body className="flex h-dvh flex-col overflow-hidden">
        {children}
        <Toaster position="top-center" />
      </body>
    </html>
  );
}
