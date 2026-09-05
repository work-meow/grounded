"use client";

/**
 * A small panel anchored to what opened it.
 *
 * Radix's Popover underneath, for the same reasons as the sheet: it closes on
 * Escape and on a click outside, it moves focus into the panel and back out
 * again, and it flips itself when the anchor is near an edge — which for a
 * control sitting at the bottom of the screen is not a nicety.
 */

import * as React from "react";
import { Popover as Primitive } from "radix-ui";

import { cn } from "@/lib/utils";

function Popover(props: React.ComponentProps<typeof Primitive.Root>) {
  return <Primitive.Root data-slot="popover" {...props} />;
}

function PopoverTrigger(props: React.ComponentProps<typeof Primitive.Trigger>) {
  return <Primitive.Trigger data-slot="popover-trigger" {...props} />;
}

function PopoverContent({
  className,
  align = "start",
  sideOffset = 8,
  ...props
}: React.ComponentProps<typeof Primitive.Content>) {
  return (
    <Primitive.Portal>
      <Primitive.Content
        data-slot="popover-content"
        align={align}
        sideOffset={sideOffset}
        className={cn(
          "z-50 w-72 rounded-lg border bg-popover p-1 text-popover-foreground shadow-md outline-none",
          "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
          "data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95",
          className,
        )}
        {...props}
      />
    </Primitive.Portal>
  );
}

export { Popover, PopoverContent, PopoverTrigger };
