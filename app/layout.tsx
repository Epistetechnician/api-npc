import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { validateEnvironmentVariables } from '@/utils/env';

const inter = Inter({ subsets: ["latin"] });

// Validate environment variables at build time
validateEnvironmentVariables();

export const metadata: Metadata = {
  title: "AIAPICaba!",
  description: "Advanced terminal interface for exploring and analyzing HyperLiquid API data",
  icons: {
    icon: "/favicon.ico",
  },
  viewport: "width=device-width, initial-scale=1",
  keywords: "API, blockchain, data analytics, terminal",
  authors: [{ name: "npc Team" }],
  openGraph: {
    title: "Terminal",
    description: "Advanced terminal interface for exploring and analyzing API data",
    type: "website",
  }
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
