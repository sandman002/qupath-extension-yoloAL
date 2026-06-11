package qupath.ext.yoloal

import com.google.gson.Gson
import qupath.lib.images.ImageData
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.interfaces.ROI

import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Random
import java.util.concurrent.atomic.AtomicBoolean
import java.util.logging.Logger

/**
 * Extracts YOLO-format image patches from QuPath annotations.
 *
 * Scale-aware: for each annotation the downsample factor is chosen so the
 * annotation bounding box fills ~targetFill of the tile.  All output images
 * are patchSize × patchSize pixels regardless of the downsample used.
 *
 * Cancel support: set cancelFlag to the panel's AtomicBoolean before calling extract().
 * Filenames use annotation UUID for stable identity across AL cycles.
 * Origin sidecar JSONs let inference map patch→slide coordinates.
 */
class PatchExtractor {

    static final Logger log = Logger.getLogger(PatchExtractor.class.name)

    int    patchSize   = 640
    int    maxOffset   = 100
    int    augPerBox   = 1
    String datasetRoot
    String split       = "train"

    /**
     * Annotation bbox should fill this fraction of the tile side length.
     * E.g. 0.66 → a 640px tile covers ~960 full-res pixels for a 640px object.
     */
    double targetFill  = 0.66

    /**
     * Allowed downsample values to snap to.
     * neededDs is rounded UP to the nearest entry so the annotation always fits.
     */
    List<Double> allowedDownsamples = [1.0, 2.0, 4.0, 8.0]

    /** If set, extract() checks this after every patch and stops when true. */
    AtomicBoolean cancelFlag = null

    /** Optional progress callback: closure(int done, int expected). */
    Closure onPatchProgress = null

    private final Random rng  = new Random()
    private final Gson   gson = new Gson()

    /**
     * Extract patches for one slide.
     * @return Map {total: int, scaleStats: {"1×": N, "2×": N, …}}
     */
    Map extract(ImageData<BufferedImage> imageData, String slideId) {

        def server     = imageData.getServer()
        def serverPath = server.getPath()
        def hierarchy  = imageData.getHierarchy()

        List<PathObject> annotations = hierarchy.getAnnotationObjects()
            .findAll { it.getROI() != null && it.getPathClass() != null && it.getPathClass().getName() != null }

        if (annotations.isEmpty()) {
            log.warning("No classified annotations on: $slideId")
            return [total: 0, scaleStats: [:]]
        }

        List<String> classNames = annotations.collect { it.getPathClass().getName() }.unique().sort()

        Path imgDir  = Paths.get(datasetRoot, "images",  split)
        Path lblDir  = Paths.get(datasetRoot, "labels",  split)
        Path origDir = Paths.get(datasetRoot, "origins", split)
        [imgDir, lblDir, origDir].each { Files.createDirectories(it) }

        int expected = annotations.size() * augPerBox
        int done     = 0
        Map<String, Integer> scaleStats = [:]

        for (int ai = 0; ai < annotations.size(); ai++) {
            if (cancelFlag?.get()) break

            PathObject anchor = annotations[ai]
            ROI    roi   = anchor.getROI()
            double cx    = roi.getCentroidX()
            double cy    = roi.getCentroidY()
            String annId = anchor.getID().toString().replace("-", "").substring(0, 8)

            // ── Pick downsample so the annotation fills ~targetFill of the tile ──
            double annSize  = Math.max(roi.getBoundsWidth(), roi.getBoundsHeight())
            double neededDs = Math.max(1.0, annSize / (patchSize * targetFill))
            double ds = allowedDownsamples.find { it >= neededDs } ?: allowedDownsamples.last()
            long physicalPatch = Math.round(patchSize * ds) as long  // region size in full-res pixels
            String scaleLabel  = ds == Math.floor(ds) ? "${(int) ds}×" : "${ds}×"

            for (int augIdx = 0; augIdx < augPerBox; augIdx++) {
                if (cancelFlag?.get()) break

                String stem    = "${slideId}_${annId}_aug${augIdx}"
                def    imgFile = imgDir.resolve("${stem}.png").toFile()
                def    lblFile = lblDir.resolve("${stem}.txt").toFile()
                def    oFile   = origDir.resolve("${stem}.json").toFile()

                done++
                if (onPatchProgress) onPatchProgress(done, expected)

                if (imgFile.exists() && lblFile.exists() && oFile.exists()) continue

                int dx = augIdx == 0 ? 0 : rng.nextInt(2 * maxOffset + 1) - maxOffset
                int dy = augIdx == 0 ? 0 : rng.nextInt(2 * maxOffset + 1) - maxOffset

                long x0 = Math.round(cx + dx - physicalPatch / 2.0) as long
                long y0 = Math.round(cy + dy - physicalPatch / 2.0) as long
                x0 = Math.max(0, Math.min(x0, server.getWidth()  - physicalPatch))
                y0 = Math.max(0, Math.min(y0, server.getHeight() - physicalPatch))

                RegionRequest request = RegionRequest.createInstance(
                    serverPath, ds, (int) x0, (int) y0, (int) physicalPatch, (int) physicalPatch)

                BufferedImage patch
                try { patch = server.readRegion(request) }
                catch (Exception e) { log.warning("Failed to read region: $e.message"); continue }

                // Collect all annotations whose centroid falls inside this physical region
                List<String> yoloLines = []
                for (PathObject ann : annotations) {
                    ROI    r   = ann.getROI()
                    double bcx = r.getCentroidX()
                    double bcy = r.getCentroidY()
                    if (bcx >= x0 && bcx < x0 + physicalPatch &&
                        bcy >= y0 && bcy < y0 + physicalPatch) {
                        int    ci  = classNames.indexOf(ann.getPathClass().getName())
                        // Normalize by physicalPatch (full-res region size) → [0,1] in output image
                        double cxl = (bcx - x0) / physicalPatch
                        double cyl = (bcy - y0) / physicalPatch
                        double wn  = Math.min(r.getBoundsWidth(),  physicalPatch) / physicalPatch
                        double hn  = Math.min(r.getBoundsHeight(), physicalPatch) / physicalPatch
                        yoloLines << String.format("%d %.6f %.6f %.6f %.6f", ci, cxl, cyl, wn, hn)
                    }
                }

                if (yoloLines.isEmpty()) continue

                ImageIO.write(patch, "PNG", imgFile)
                Files.write(lblFile.toPath(), yoloLines.join("\n").getBytes("UTF-8"))
                Files.write(oFile.toPath(), gson.toJson([
                    x0: x0, y0: y0, downsample: ds,
                    physicalPatch: physicalPatch,
                    slideId: slideId, patchSize: patchSize
                ]).getBytes("UTF-8"))

                scaleStats[scaleLabel] = (scaleStats[scaleLabel] ?: 0) + 1
            }
        }

        log.info("Extracted ${done} patches for $slideId  scales=${scaleStats}  split=${split}")
        return [total: done, scaleStats: scaleStats]
    }

    static void writeDatasetYaml(String datasetRoot, List<String> classNames) {
        def safeNames = classNames.findAll { it != null }
        def rootFwd = datasetRoot.replace('\\', '/')
        def hasVal  = new File(datasetRoot, "images/val").exists()
        def hasTest = new File(datasetRoot, "images/test").exists()
        def content = """\
path: ${rootFwd}
train: images/train
val:   ${hasVal  ? 'images/val'   : 'images/train'}
test:  ${hasTest ? 'images/test'  : 'images/train'}

nc: ${safeNames.size()}
names: [${safeNames.collect { "\"$it\"" }.join(", ")}]
"""
        Files.write(Paths.get(datasetRoot, "dataset.yaml"), content.getBytes("UTF-8"))
    }
}
