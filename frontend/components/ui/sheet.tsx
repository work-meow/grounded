"use client";

/**
 * A panel that slides in from the edge.
 *
 * Radix's Dialog underneath, which is what makes it a real one: focus is
 * trapped and restored, Escape closes it, the page behind stops scrolling, and
 * the rest of the document is hidden from screen readers. All of that is easy
 * to leave out of a hand-rolled drawer and impossible to notice missing.
 */

import * as React from "react";
import { X } from "lucide-react";
import { Dialog } from "radix-ui";

import { cn } from "@/lib/utils";

function Sheet(props: React.ComponentProps<typeof Dialog.Root>) {
  return <Dialog.Root data-slot="sheet" {...props} />;
}

function SheetTrigger(props: React.ComponentProps<typeof Dialog.Trigger>) {
  return <Dialog.Trigger data-slot="sheet-trigger" {...props} />;
}

function SheetContent({
  className,
  children,
  ...props
}: React.ComponentProps<typeof Dialog.Content>) {
  return (
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-50 bg-black/50 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
      <Dialog.Content
        data-slot="sheet-content"
        // Radix asks for a description or an explicit opt-out, and warns in the
        // console when it gets neither. A chat list has nothing to describe.
        aria-describedby={undefined}
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-72 max-w-[85vw] flex-col border-r bg-background shadow-lg",
          "data-[state=closed]:animate-out data-[state=closed]:slide-out-to-left data-[state=closed]:duration-200",
          "data-[state=open]:animate-in data-[state=open]:slide-in-from-left data-[state=open]:duration-300",
          className,
        )}
        {...props}
      >
        {children}
        <Dialog.Close className="absolute top-3 right-3 rounded-sm opacity-60 transition-opacity hover:opacity-100 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none">
          <X className="size-4" />
          <span className="sr-only">Закрыть</span>
        </Dialog.Close>
      </Dialog.Content>
    </Dialog.Portal>
  );
}

function SheetTitle({ className, ...props }: React.ComponentProps<typeof Dialog.Title>) {
  return (
    <Dialog.Title
      data-slot="sheet-title"
      className={cn("text-sm font-medium", className)}
      {...props}
    />
  );
}

export { Sheet, SheetContent, SheetTitle, SheetTrigger };
