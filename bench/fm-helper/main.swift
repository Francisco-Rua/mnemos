// Tiny bridge to Apple's on-device Foundation Model (macOS 26+, Apple Intelligence on), for the local-mode
// benchmark only. Reads one JSON object per line on stdin {"prompt": "..."}, writes one JSON line per request:
// {"text": "..."} or {"error": "..."}. First line written is {"available": true|false, "reason": "..."}.
import Foundation
import FoundationModels

func emit(_ o: [String: Any]) {
    let d = try! JSONSerialization.data(withJSONObject: o)
    FileHandle.standardOutput.write(d)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

@main
struct FMHelper {
    static func main() async {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            emit(["available": true, "reason": ""])
        default:
            emit(["available": false, "reason": String(describing: model.availability)])
            return
        }
        while let line = readLine(strippingNewline: true) {
            guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let prompt = obj["prompt"] as? String else { emit(["error": "bad request"]); continue }
            do {
                let session = LanguageModelSession(instructions: obj["instructions"] as? String ?? "")
                let opts = GenerationOptions(temperature: 0)
                let r = try await session.respond(to: prompt, options: opts)
                emit(["text": r.content])
            } catch {
                emit(["error": String(describing: error)])
            }
        }
    }
}
