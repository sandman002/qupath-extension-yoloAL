package qupath.ext.yoloal

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.util.logging.Logger

/**
 * Thin HTTP client for the local Python YOLO server (default: http://localhost:5005).
 * Uses Gson (shipped with QuPath) instead of groovy-json to avoid classloader conflicts.
 * All methods are blocking — call from a background thread.
 */
class ServerClient {

    static final Logger log = Logger.getLogger(ServerClient.class.name)

    String baseUrl = "http://localhost:5005"
    private final HttpClient http = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .build()
    private final Gson gson = new Gson()
    private final java.lang.reflect.Type mapType = new TypeToken<Map<String, Object>>(){}.getType()

    // ── server lifecycle ───────────────────────────────────────────────────

    /** Returns the compute device string, e.g. "GPU — RTX 4090 (16.0 GB)" or "CPU". */
    String getDevice() {
        try { return (get("/device")?.get("device") as String) ?: "unknown" }
        catch (Exception e) { log.warning("getDevice failed: $e.message"); return "unknown" }
    }

    /** Returns true if server responds to /ping within 2 s. */
    boolean isAlive() {
        try {
            def req = HttpRequest.newBuilder()
                .uri(URI.create("${baseUrl}/ping"))
                .timeout(java.time.Duration.ofSeconds(2))
                .GET()
                .build()
            def resp = http.send(req, HttpResponse.BodyHandlers.ofString())
            return resp.statusCode() == 200
        } catch (Exception ignored) {
            return false
        }
    }

    /** Ask the server to shut itself down gracefully, then wait up to 3 s for it to stop. */
    void shutdown() {
        try { post("/shutdown", [:]) } catch (Exception ignored) {}
        int tries = 0
        while (isAlive() && tries++ < 6) Thread.sleep(500)
    }

    /**
     * Start the Python server as a subprocess.
     * @param pythonExe    path to python executable (inside venv)
     * @param serverScript path to yolo_server.py
     * @param port         port number
     */
    Process startServer(String pythonExe, String serverScript, int port = 5005, File workDir = null) {
        baseUrl = "http://localhost:${port}"
        def cmd = [pythonExe, serverScript, "--port", port.toString()]
        log.info("Starting server: ${cmd.join(' ')}")
        def pb = new ProcessBuilder(cmd)
        pb.redirectErrorStream(true)
        // Set working dir so YOLO's runs/ folder stays inside our project root, not QuPath's app dir
        if (workDir?.exists()) pb.directory(workDir)
        return pb.start()
    }

    // ── dataset ────────────────────────────────────────────────────────────

    Map setDataset(String yamlPath) { post("/dataset", [yaml_path: yamlPath]) }

    // ── training ───────────────────────────────────────────────────────────

    Map startTraining(Map config) { post("/train", config) }

    Map stopTraining() { post("/stop", [:]) }

    /** Poll while training. Returns: {state, epoch, epochs, train_loss, val_loss, map50, map50_95} */
    Map getStatus() { get("/status") }

    /** Final eval metrics after training finishes. Saves val run into saveDir if provided. */
    Map getMetrics(String saveDir = null) {
        def body = saveDir ? [save_dir: saveDir] : [:]
        post("/metrics", body)
    }

    // ── inference ─────────────────────────────────────────────────────────

    /**
     * Detect tissue bounding box in a thumbnail.
     * @param base64Png  base64-encoded PNG of the thumbnail
     * @param thumbW/H   thumbnail dimensions
     * @param fullW/H    full-resolution slide dimensions
     * @return list of [x0,y0,x1,y1] rectangles in full-resolution coords
     */
    List getTissueBounds(String base64Png, int thumbW, int thumbH, int fullW, int fullH) {
        def res = post("/tissue", [
            image_base64: base64Png,
            thumb_w: thumbW, thumb_h: thumbH,
            full_w: fullW,   full_h: fullH
        ])
        return res?.get("bounds") as List ?: [[0, 0, fullW, fullH]]
    }

    List runInference(List<String> imagePaths, double conf = 0.25, double iou = 0.45, String modelPath = null) {
        def body = [image_paths: imagePaths, conf: conf, iou: iou]
        if (modelPath) body.model_path = modelPath
        def res = post("/infer", body, 300)
        return res?.get("results") as List ?: []
    }

    /**
     * Full WSI inference via Python (tiling + YOLO + NMS + GeoJSON).
     * params keys: slide_path, model_path, conf, iou, tile_size, overlap,
     *              tissue_frac, output_dir, roi_bounds (list of {x,y,w,h})
     * Returns map with: detections, tiles_total, tiles_used, geojson_path, error
     */
    Map inferWSI(Map params) { post("/infer_wsi", params, 1800) }

    /** Poll during WSI inference. Returns: {active, done, total, dets, slide} */
    Map getInferProgress() { get("/infer_progress") }

    // ── private helpers ────────────────────────────────────────────────────

    private Map get(String path, int timeoutSecs = 10) {
        def req = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl}${path}"))
            .timeout(java.time.Duration.ofSeconds(timeoutSecs))
            .GET()
            .build()
        def resp = http.send(req, HttpResponse.BodyHandlers.ofString())
        try {
            return gson.fromJson(resp.body(), mapType) as Map
        } catch (Exception e) {
            def snippet = (resp.body() ?: "").take(300)
            return [error: "HTTP ${resp.statusCode()} — ${snippet}"] as Map
        }
    }

    private Map post(String path, Map body, int timeoutSecs = 30) {
        def json = gson.toJson(body)
        def req = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl}${path}"))
            .timeout(java.time.Duration.ofSeconds(timeoutSecs))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .build()
        def resp = http.send(req, HttpResponse.BodyHandlers.ofString())
        try {
            return gson.fromJson(resp.body(), mapType) as Map
        } catch (Exception e) {
            def snippet = (resp.body() ?: "").take(300)
            return [error: "HTTP ${resp.statusCode()} — ${snippet}"] as Map
        }
    }
}
