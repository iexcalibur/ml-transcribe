# TinyLlamaDemo — running our Swift package on a real iPhone

Tiny SwiftUI app that loads TinyLlama-1.1B from the app bundle and generates
text from a prompt. Built to verify on physical iPhone hardware that the
F16 storage path actually fits in iOS's per-app memory budget.

What's here:

```
TinyLlamaDemoApp.swift     // @main App entry
ContentView.swift          // SwiftUI UI: prompt + Generate button + output
TinyLlamaGenerator.swift   // Loads from bundle, runs DecoderLM, decodes
Info.plist                 // Minimal iOS app metadata
```

Not committed (you supply these — see "Adding the model" below):

```
model.safetensors          // 2.2 GB
tokenizer.json             // ~1.8 MB
```

## Setup (5–10 minutes)

The fastest path is to let Xcode generate the project:

1. **Open Xcode** → File → New → Project → iOS → App.
2. Name it `TinyLlamaDemo`. Interface: **SwiftUI**. Language: **Swift**.
3. Save into `apps/` (next to this folder, NOT inside it).
4. In the new project, **delete** the auto-generated `ContentView.swift` and
   `TinyLlamaDemoApp.swift`.
5. **Drag** the four files from this folder
   (`TinyLlamaDemoApp.swift`, `ContentView.swift`,
   `TinyLlamaGenerator.swift`, `Info.plist`) into the Xcode project's
   sidebar. When prompted, **uncheck** "Copy items if needed" so they
   stay version-controlled in this repo.
6. **Add the Swift Package**: File → Add Package Dependencies → "Add
   Local…" → pick `swift/` from this repo. Add the `Transcribe`
   product to the app target.

That's it for the project skeleton. Build it once for the simulator
to confirm everything's wired:

```
xcodebuild -project apps/TinyLlamaDemo.xcodeproj \
           -scheme TinyLlamaDemo \
           -destination 'platform=iOS Simulator,name=iPhone 15' \
           build
```

(Or hit ⌘B in Xcode.)

## Adding the model

The app loads `model.safetensors` and `tokenizer.json` from the
**bundle resources**. The model file is 2.2 GB so we don't commit it.

1. Download both files to `/tmp/ml-inspect/tinyllama/` (or anywhere
   you like):

   ```sh
   mkdir -p /tmp/ml-inspect/tinyllama && cd /tmp/ml-inspect/tinyllama
   curl -L -O https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
   curl -L -O https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json
   ```

2. **In Xcode**, drag both files into the project's resources, with
   the app target checked. Confirm they show up under "Copy Bundle
   Resources" in the target's Build Phases.

The 2.2 GB will inflate your app build to ~2.5 GB. That's fine for
sideloading to your own device but won't fit through App Store
review (4 GB binary cap, but bundled-data must compress).

## Deploying to a physical iPhone

Prerequisites:
- An Apple ID (a free Apple Developer account is sufficient).
- iPhone 15 / iPhone 15 Pro / iPhone 16 family. Older iPhones with
  ≤ 6 GB RAM may hit Jetsam during the 2.2 GB load.

Steps:

1. **Plug your iPhone into your Mac** via USB, unlock it, trust
   the computer when prompted.
2. **In Xcode**, click the project in the sidebar → Signing &
   Capabilities → check "Automatically manage signing", select
   your Apple ID team. Choose any unique bundle identifier.
3. **Select your iPhone** as the run destination (top toolbar
   device dropdown).
4. **Run** (⌘R). First install takes a minute (the 2.5 GB binary
   has to push over USB).
5. On the iPhone, the first launch may prompt you to **trust the
   developer certificate** under Settings → General → VPN & Device
   Management.
6. Enter a prompt, tap **Generate**. Expect:
   - ~10–20 seconds for the first model load (one-time).
   - ~0.3–0.5 seconds per token after that.

## What we expect to see (memory)

TinyLlama-1.1B in F16:

```
Weights at rest:        ~2.2 GB
Activations + KV cache: ~100 MB (max-seq=256, 22 layers)
Total resident:         ~2.4 GB
```

iPhone per-app memory budgets (rough):

| Device | Practical limit before Jetsam |
|---|---|
| iPhone 15 / 16 (8 GB RAM) | ~4–5 GB |
| iPhone 15 Pro / 16 Pro (8 GB RAM) | ~4–5 GB |
| iPhone 14 / 13 (6 GB RAM) | ~3 GB |
| iPhone 12 / older | likely won't fit |

So we expect this to work on iPhone 15 / 16 family. Older iPhones
are an open question.

## What this demonstrates

If the app loads + generates without being killed by Jetsam, it
proves end-to-end that:

1. The Swift package's xcframework links cleanly into a real iOS
   app target (not just XCTest).
2. The F16 storage + NEON SIMD matmul path works on iOS arm64.
3. A 1.1B-parameter pretrained LLM fits in iOS's per-app memory
   budget on modern hardware.
4. Inference runs at "interactive but not real-time" speed
   (~0.3–0.5 sec/token on Apple A17/A18, no Metal acceleration).
