import Foundation
import AVFoundation
import Vision

public struct RPPGSystemConfig {
    public var windowSize: TimeInterval
    public var qualityThreshold: Double
    public var enableAdaptiveProcessing: Bool
    public var targetFPS: Double

    public init(
        windowSize: TimeInterval = RPPGConstants.windowSize,
        qualityThreshold: Double = RPPGConstants.minConfidence,
        enableAdaptiveProcessing: Bool = true,
        targetFPS: Double = RPPGConstants.defaultFPS
    ) {
        self.windowSize = windowSize
        self.qualityThreshold = qualityThreshold
        self.enableAdaptiveProcessing = enableAdaptiveProcessing
        self.targetFPS = targetFPS
    }
}

struct ReplayConfig {
    var videoPath: String = ""
    var outputCSV: String = ""
    var maxSeconds: Int = -1
    var preset: String = "robustMode"
    var roiDebugCSV: String = ""
    var algoDebugCSV: String = ""
}

struct SecondRow {
    var sec: Int
    var hrBest: Double?
    var hrPublished: Double?
    var confidence: Double?
    var signalQuality: Double?
    var roiCount: Int
    var faceConfidence: Float?
    var frameCount: Int
}

struct ROIDebugRow {
    var frameIndex: Int
    var sec: Int
    var regionName: String
    var roiX: CGFloat
    var roiY: CGFloat
    var roiW: CGFloat
    var roiH: CGFloat
    var roiWeight: Float
    var faceX: CGFloat?
    var faceY: CGFloat?
    var faceW: CGFloat?
    var faceH: CGFloat?
    var faceConfidence: Float?
    var trackingMode: String
    var roiCountInFrame: Int
}

struct AlgoDebugRow {
    var frameIndex: Int
    var sec: Int
    var hrBest: Double?
    var hrPublished: Double?
    var rawFusedBPM: Double?
    var constrainedBPM: Double?
    var assistedBPM: Double?
    var fusionBestBPM: Double?
    var ppiHeartRate: Double?
    var frequencyConfidence: Double?
    var sqi: Double?
    var signalQuality: Double?
    var confidence: Double?
    var readyROICount: Int?
    var candidateCount: Int?
    var candidates: String
}

func parseArgs() -> ReplayConfig {
    var cfg = ReplayConfig()
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        let key = args[i]
        if key == "--video-path", i + 1 < args.count {
            cfg.videoPath = args[i + 1]
            i += 2
            continue
        }
        if key == "--out-csv", i + 1 < args.count {
            cfg.outputCSV = args[i + 1]
            i += 2
            continue
        }
        if key == "--max-seconds", i + 1 < args.count {
            cfg.maxSeconds = Int(args[i + 1]) ?? -1
            i += 2
            continue
        }
        if key == "--preset", i + 1 < args.count {
            cfg.preset = args[i + 1]
            i += 2
            continue
        }
        if key == "--roi-debug-csv", i + 1 < args.count {
            cfg.roiDebugCSV = args[i + 1]
            i += 2
            continue
        }
        if key == "--algo-debug-csv", i + 1 < args.count {
            cfg.algoDebugCSV = args[i + 1]
            i += 2
            continue
        }
        i += 1
    }
    return cfg
}

func usageAndExit() -> Never {
    fputs(
        "usage: ios_offline_replay --video-path <path> --out-csv <path> [--max-seconds <int>] [--preset <robustMode|pythonAligned|ios26Hybrid|accuracyPriority|v4Compatible>] [--roi-debug-csv <path>] [--algo-debug-csv <path>]\n",
        stderr
    )
    exit(2)
}

func resolvePreset(_ raw: String) -> RPPGProcessingPreset {
    switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "robustmode", "robust", "default":
        return .robustMode
    case "pythonaligned", "python":
        return .pythonAligned
    case "ios26hybrid", "hybrid":
        return .ios26Hybrid
    case "accuracypriority", "accuracy":
        return .accuracyPriority
    case "v4compatible", "v4":
        return .v4Compatible
    default:
        return .robustMode
    }
}

func metadataRectToPixelRect(_ metadataRect: CGRect, pixelBuffer: CVPixelBuffer) -> CGRect {
    let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
    let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
    let minX = max(0.0, min(metadataRect.minX, 1.0))
    let maxX = min(1.0, max(metadataRect.maxX, 0.0))
    let minY = max(0.0, min(metadataRect.minY, 1.0))
    let maxY = min(1.0, max(metadataRect.maxY, 0.0))
    if maxX <= minX || maxY <= minY {
        return CGRect(x: 0.3 * width, y: 0.2 * height, width: 0.4 * width, height: 0.2 * height)
    }
    return CGRect(
        x: minX * width,
        y: minY * height,
        width: (maxX - minX) * width,
        height: (maxY - minY) * height
    )
}

func visionOrientationForOfflineFrame(_ pixelBuffer: CVPixelBuffer) -> CGImagePropertyOrientation {
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    return width >= height ? .right : .up
}

func writeCSV(rows: [SecondRow], to path: String) throws {
    let outURL = URL(fileURLWithPath: path)
    let dir = outURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    var text = "sec,hr_best,hr_published,confidence,signal_quality,roi_count,face_confidence,frame_count\n"
    for row in rows.sorted(by: { $0.sec < $1.sec }) {
        let hrBest = row.hrBest.map { String(format: "%.6f", $0) } ?? ""
        let hrPub = row.hrPublished.map { String(format: "%.6f", $0) } ?? ""
        let conf = row.confidence.map { String(format: "%.6f", $0) } ?? ""
        let sq = row.signalQuality.map { String(format: "%.6f", $0) } ?? ""
        let faceConf = row.faceConfidence.map { String(format: "%.6f", $0) } ?? ""
        text += "\(row.sec),\(hrBest),\(hrPub),\(conf),\(sq),\(row.roiCount),\(faceConf),\(row.frameCount)\n"
    }
    try text.write(to: outURL, atomically: true, encoding: .utf8)
}

func writeROIDebugCSV(rows: [ROIDebugRow], to path: String) throws {
    let outURL = URL(fileURLWithPath: path)
    let dir = outURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    var text = "frame_index,sec,region_name,roi_x,roi_y,roi_w,roi_h,roi_weight,face_x,face_y,face_w,face_h,face_confidence,tracking_mode,roi_count_in_frame\n"
    for row in rows {
        let fx = row.faceX.map { String(format: "%.6f", $0) } ?? ""
        let fy = row.faceY.map { String(format: "%.6f", $0) } ?? ""
        let fw = row.faceW.map { String(format: "%.6f", $0) } ?? ""
        let fh = row.faceH.map { String(format: "%.6f", $0) } ?? ""
        let fconf = row.faceConfidence.map { String(format: "%.6f", $0) } ?? ""
        text += "\(row.frameIndex),\(row.sec),\(row.regionName),\(String(format: "%.6f", row.roiX)),\(String(format: "%.6f", row.roiY)),\(String(format: "%.6f", row.roiW)),\(String(format: "%.6f", row.roiH)),\(String(format: "%.6f", row.roiWeight)),\(fx),\(fy),\(fw),\(fh),\(fconf),\(row.trackingMode),\(row.roiCountInFrame)\n"
    }
    try text.write(to: outURL, atomically: true, encoding: .utf8)
}

func writeAlgoDebugCSV(rows: [AlgoDebugRow], to path: String) throws {
    let outURL = URL(fileURLWithPath: path)
    let dir = outURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    var text = "frame_index,sec,hr_best,hr_published,raw_fused_bpm,constrained_bpm,assisted_bpm,fusion_best_bpm,ppi_hr,freq_confidence,sqi,signal_quality,confidence,ready_roi_count,candidate_count,candidates\n"
    for row in rows {
        let hrBest = row.hrBest.map { String(format: "%.6f", $0) } ?? ""
        let hrPublished = row.hrPublished.map { String(format: "%.6f", $0) } ?? ""
        let rawFusedBPM = row.rawFusedBPM.map { String(format: "%.6f", $0) } ?? ""
        let constrainedBPM = row.constrainedBPM.map { String(format: "%.6f", $0) } ?? ""
        let assistedBPM = row.assistedBPM.map { String(format: "%.6f", $0) } ?? ""
        let fusionBestBPM = row.fusionBestBPM.map { String(format: "%.6f", $0) } ?? ""
        let ppiHeartRate = row.ppiHeartRate.map { String(format: "%.6f", $0) } ?? ""
        let frequencyConfidence = row.frequencyConfidence.map { String(format: "%.6f", $0) } ?? ""
        let sqi = row.sqi.map { String(format: "%.6f", $0) } ?? ""
        let signalQuality = row.signalQuality.map { String(format: "%.6f", $0) } ?? ""
        let confidence = row.confidence.map { String(format: "%.6f", $0) } ?? ""
        let readyROICount = row.readyROICount.map(String.init) ?? ""
        let candidateCount = row.candidateCount.map(String.init) ?? ""
        let candidates = row.candidates.replacingOccurrences(of: ",", with: ";")
        text += "\(row.frameIndex),\(row.sec),\(hrBest),\(hrPublished),\(rawFusedBPM),\(constrainedBPM),\(assistedBPM),\(fusionBestBPM),\(ppiHeartRate),\(frequencyConfidence),\(sqi),\(signalQuality),\(confidence),\(readyROICount),\(candidateCount),\(candidates)\n"
    }
    try text.write(to: outURL, atomically: true, encoding: .utf8)
}

func readerStatusText(_ status: AVAssetReader.Status) -> String {
    switch status {
    case .unknown:
        return "unknown"
    case .reading:
        return "reading"
    case .completed:
        return "completed"
    case .failed:
        return "failed"
    case .cancelled:
        return "cancelled"
    @unknown default:
        return "unknown_default"
    }
}

func buildTrackOutputReader(
    asset: AVAsset,
    track: AVAssetTrack,
    outputSettings: [String: Any]
) -> (AVAssetReader, AVAssetReaderOutput)? {
    do {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            return nil
        }
        reader.add(output)
        guard reader.startReading() else {
            let errText = reader.error.map { String(describing: $0) } ?? "nil"
            fputs(
                "AVAssetReader(trackOutput) failed to start. status=\(readerStatusText(reader.status)) error=\(errText)\n",
                stderr
            )
            return nil
        }
        return (reader, output)
    } catch {
        fputs("failed to create AVAssetReader(trackOutput): \(error)\n", stderr)
        return nil
    }
}

func buildVideoCompositionReader(
    asset: AVAsset,
    track: AVAssetTrack,
    outputSettings: [String: Any]
) -> (AVAssetReader, AVAssetReaderOutput)? {
    do {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderVideoCompositionOutput(videoTracks: [track], videoSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        output.videoComposition = AVMutableVideoComposition(propertiesOf: asset)
        guard reader.canAdd(output) else {
            return nil
        }
        reader.add(output)
        guard reader.startReading() else {
            let errText = reader.error.map { String(describing: $0) } ?? "nil"
            fputs(
                "AVAssetReader(videoCompositionOutput) failed to start. status=\(readerStatusText(reader.status)) error=\(errText)\n",
                stderr
            )
            return nil
        }
        return (reader, output)
    } catch {
        fputs("failed to create AVAssetReader(videoCompositionOutput): \(error)\n", stderr)
        return nil
    }
}

@main
struct IOSOfflineReplayMain {
    static func main() async {
        let cfg = parseArgs()
        if cfg.videoPath.isEmpty || cfg.outputCSV.isEmpty {
            usageAndExit()
        }
        if ProcessInfo.processInfo.environment["IOS_REPLAY_SILENT"] == "1" {
            _ = freopen("/dev/null", "w", stdout)
        }

        RPPGConstants.currentPreset = resolvePreset(cfg.preset)
        RPPGConstants.enableLogging = false
        RPPGConstants.enableVerboseLogging = false
        UserDefaults.standard.set(false, forKey: "developerModeEnabled")

        let videoURL = URL(fileURLWithPath: cfg.videoPath)
        let asset = AVURLAsset(url: videoURL)
        let isPlayable = (try? await asset.load(.isPlayable)) ?? false
        let isReadable = (try? await asset.load(.isReadable)) ?? false
        if !isPlayable || !isReadable {
            fputs(
                "asset capability warning: playable=\(isPlayable) readable=\(isReadable) path=\(cfg.videoPath)\n",
                stderr
            )
        }

        let track: AVAssetTrack
        do {
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let first = tracks.first else {
                fputs("no video track found: \(cfg.videoPath)\n", stderr)
                exit(1)
            }
            track = first
        } catch {
            fputs("failed to load video track: \(cfg.videoPath) error=\(error)\n", stderr)
            exit(1)
        }
        let loadedNominalFrameRate = (try? await track.load(.nominalFrameRate)) ?? 0.0
        let nominalFPS = {
            let fps = Double(loadedNominalFrameRate)
            return fps.isFinite && fps >= 10.0 ? fps : 30.0
        }()

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        ]
        let setup = buildTrackOutputReader(asset: asset, track: track, outputSettings: outputSettings)
            ?? buildVideoCompositionReader(asset: asset, track: track, outputSettings: outputSettings)
        guard let (reader, output) = setup else {
            fputs("AVAssetReader failed to start for all output types\n", stderr)
            exit(1)
        }

        let tracker = FaceROITracker()
        let processor = RPPGProcessor()
        await processor.startProcessing()
        processor.updateSamplingRate(nominalFPS)

        var rowsBySec: [Int: SecondRow] = [:]
        var roiDebugRows: [ROIDebugRow] = []
        var algoDebugRows: [AlgoDebugRow] = []
        var totalFrames = 0
        while let sampleBuffer = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                continue
            }
            let ts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let secFloat = CMTimeGetSeconds(ts)
            if !secFloat.isFinite || secFloat < 0 {
                continue
            }
            let sec = Int(floor(secFloat))
            if cfg.maxSeconds >= 0 && sec > cfg.maxSeconds {
                break
            }

            totalFrames += 1
            var row = rowsBySec[sec] ?? SecondRow(
                sec: sec,
                hrBest: nil,
                hrPublished: nil,
                confidence: nil,
                signalQuality: nil,
                roiCount: 0,
                faceConfidence: nil,
                frameCount: 0
            )
            row.frameCount += 1

            let orientation = visionOrientationForOfflineFrame(pixelBuffer)
            let (trackingResult, rois) = await tracker.processFrame(pixelBuffer, orientation: orientation)
            if let trackingResult {
                row.faceConfidence = trackingResult.confidence
            }
            guard let rois, !rois.isEmpty else {
                rowsBySec[sec] = row
                continue
            }

            if !cfg.roiDebugCSV.isEmpty {
                let faceRect = trackingResult?.faceBounds
                let trackingMode: String
                if let m = trackingResult?.mode {
                    switch m {
                    case .detection: trackingMode = "detection"
                    case .tracking: trackingMode = "tracking"
                    case .redetection: trackingMode = "redetection"
                    }
                } else {
                    trackingMode = "unknown"
                }
                for item in rois {
                    roiDebugRows.append(
                        ROIDebugRow(
                            frameIndex: totalFrames,
                            sec: sec,
                            regionName: item.regionName,
                            roiX: item.roi.origin.x,
                            roiY: item.roi.origin.y,
                            roiW: item.roi.width,
                            roiH: item.roi.height,
                            roiWeight: item.weight,
                            faceX: faceRect?.origin.x,
                            faceY: faceRect?.origin.y,
                            faceW: faceRect?.width,
                            faceH: faceRect?.height,
                            faceConfidence: trackingResult?.confidence,
                            trackingMode: trackingMode,
                            roiCountInFrame: rois.count
                        )
                    )
                }
            }

            row.roiCount = max(row.roiCount, rois.count)
            let pixelROIs: [WeightedROI] = rois.map {
                WeightedROI(
                    roi: $0.roi,
                    weight: $0.weight,
                    regionName: $0.regionName
                )
            }

            if let result = await processor.processMultiROIFrame(pixelBuffer, weightedROIs: pixelROIs) {
                row.hrBest = result.heartRate
                row.confidence = result.confidence
                row.signalQuality = result.signalQuality
                row.hrPublished = await MainActor.run { processor.currentHeartRate }
                if !cfg.algoDebugCSV.isEmpty {
                    let fusionDebug = processor.latestFusionCycleDebugSnapshot()
                    let candidates = fusionDebug?.candidates
                        .map { "\($0.roiName):\($0.algorithm):\(String(format: "%.2f", $0.bpm))" }
                        .joined(separator: "|") ?? ""
                    algoDebugRows.append(
                        AlgoDebugRow(
                            frameIndex: totalFrames,
                            sec: sec,
                            hrBest: result.heartRate,
                            hrPublished: row.hrPublished,
                            rawFusedBPM: fusionDebug?.rawFusedBPM,
                            constrainedBPM: fusionDebug?.constrainedBPM,
                            assistedBPM: fusionDebug?.assistedBPM,
                            fusionBestBPM: fusionDebug?.bestBPM,
                            ppiHeartRate: fusionDebug?.ppiHeartRate,
                            frequencyConfidence: fusionDebug?.frequencyConfidence,
                            sqi: fusionDebug?.sqi,
                            signalQuality: fusionDebug?.signalQuality,
                            confidence: fusionDebug?.confidence,
                            readyROICount: fusionDebug?.readyROICount,
                            candidateCount: fusionDebug?.candidateCount,
                            candidates: candidates
                        )
                    )
                }
            }

            rowsBySec[sec] = row
        }

        do {
            let rows = rowsBySec.values.sorted(by: { $0.sec < $1.sec })
            try writeCSV(rows: rows, to: cfg.outputCSV)
            if !cfg.roiDebugCSV.isEmpty {
                try writeROIDebugCSV(rows: roiDebugRows, to: cfg.roiDebugCSV)
            }
            if !cfg.algoDebugCSV.isEmpty {
                try writeAlgoDebugCSV(rows: algoDebugRows, to: cfg.algoDebugCSV)
            }
        } catch {
            fputs("failed to write csv: \(error)\n", stderr)
            exit(1)
        }

        fputs("ios_offline_replay done: frames=\(totalFrames), secs=\(rowsBySec.count), out=\(cfg.outputCSV)\n", stderr)
    }
}
