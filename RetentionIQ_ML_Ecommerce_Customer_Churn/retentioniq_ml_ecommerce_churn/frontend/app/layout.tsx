import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "RetentionIQ — Customer Churn Intelligence",
  description: "ML-powered customer churn intelligence for e-commerce teams",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
