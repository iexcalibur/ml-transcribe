import SwiftUI
import Transcribe

struct ContentView: View {
    @StateObject private var generator = TinyLlamaGenerator()
    @State private var prompt: String = "The capital of France is"
    @State private var maxTokens: Int = 20
    @State private var output: String = ""
    @State private var isGenerating = false
    @State private var stats: String = ""

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Model status").font(.headline)
                Text(generator.status)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                Divider()

                Text("Prompt").font(.headline)
                TextField("prompt", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...6)

                HStack {
                    Stepper("max new tokens: \(maxTokens)",
                            value: $maxTokens, in: 1...100, step: 1)
                }

                Button {
                    Task { await generate() }
                } label: {
                    HStack {
                        if isGenerating {
                            ProgressView().controlSize(.small)
                            Text("generating…")
                        } else {
                            Image(systemName: "play.fill")
                            Text("Generate")
                        }
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerating || !generator.isReady)

                Divider()

                Text("Output").font(.headline)
                ScrollView {
                    Text(output.isEmpty ? "(no output yet)" : output)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: .infinity)

                if !stats.isEmpty {
                    Text(stats)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .navigationTitle("TinyLlama on Swift")
            .task {
                await generator.loadIfNeeded()
            }
        }
    }

    private func generate() async {
        isGenerating = true
        output = ""
        stats = ""
        defer { isGenerating = false }
        do {
            let start = Date()
            let result = try await generator.generate(
                prompt: prompt, maxNewTokens: maxTokens
            )
            let elapsed = Date().timeIntervalSince(start)
            output = result
            stats = String(
                format: "%d tokens in %.1f s (%.2f tok/s)",
                maxTokens, elapsed, Double(maxTokens) / elapsed
            )
        } catch {
            output = "ERROR: \(error)"
        }
    }
}
