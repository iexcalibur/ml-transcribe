// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Transcribe",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "Transcribe", targets: ["Transcribe"]),
    ],
    targets: [
        // Binary xcframework wrapping the Rust staticlib.
        // Built by: scripts/build-swift-ffi.sh
        .binaryTarget(
            name: "TranscribeFFIC",
            path: "artifacts/TranscribeFFI.xcframework"
        ),
        // Swift-idiomatic wrapper over the raw C symbols.
        .target(
            name: "Transcribe",
            dependencies: ["TranscribeFFIC"],
            path: "Sources/Transcribe"
        ),
        .testTarget(
            name: "TranscribeTests",
            dependencies: ["Transcribe"],
            path: "Tests/TranscribeTests"
        ),
    ]
)
