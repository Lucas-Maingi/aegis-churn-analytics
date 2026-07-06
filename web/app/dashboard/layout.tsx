"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { clearSession, getOrgName, getToken } from "@/lib/api";

const NAV = [
  { href: "/dashboard", label: "Customers", icon: "👥" },
  { href: "/dashboard/upload", label: "Import data", icon: "📤" },
  { href: "/dashboard/outreach", label: "Outreach log", icon: "✉️" },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [orgName, setOrgName] = useState("");

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    setOrgName(getOrgName());
  }, [router]);

  function logout() {
    clearSession();
    router.replace("/login");
  }

  return (
    <div className="flex min-h-screen">
      <aside className="flex w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-900/60 px-4 py-6">
        <div className="mb-1 px-2 text-xl font-bold tracking-tight">
          <span className="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
            Aegis
          </span>
        </div>
        <p className="mb-8 truncate px-2 text-xs text-slate-500">{orgName}</p>

        <nav className="flex flex-col gap-1">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`rounded-lg px-3 py-2 text-sm transition ${
                pathname === item.href
                  ? "bg-sky-500/15 font-semibold text-sky-300"
                  : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              <span className="mr-2">{item.icon}</span>
              {item.label}
            </Link>
          ))}
        </nav>

        <button
          onClick={logout}
          className="mt-auto rounded-lg px-3 py-2 text-left text-sm text-slate-400 transition hover:bg-slate-800 hover:text-slate-200"
        >
          ← Sign out
        </button>
      </aside>

      <main className="min-w-0 flex-1 px-8 py-8">{children}</main>
    </div>
  );
}
