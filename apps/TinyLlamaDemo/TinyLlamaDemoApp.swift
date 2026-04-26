import SwiftUI

/// SwiftUI sample app that loads TinyLlama-1.1B from the bundle and
/// generates text from a prompt — purely for demonstrating that the
/// `Transcribe` Swift package runs the full pipeline on real iOS
/// hardware.
///
/// To run on a physical iPhone, see `README.md` in this directory.
@main
struct TinyLlamaDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
