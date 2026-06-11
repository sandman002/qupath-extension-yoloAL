package qupath.ext.yoloal

import javafx.application.Platform
import javafx.stage.Stage
import javafx.scene.Scene
import qupath.lib.gui.QuPathGUI
import qupath.ext.yoloal.ui.YoloALPanel

/**
 * Opens (or brings to front) the YOLO AL floating window.
 */
class YoloALCommand {

    private final QuPathGUI qupath
    private static Stage stage

    YoloALCommand(QuPathGUI qupath) {
        this.qupath = qupath
    }

    void show() {
        Platform.runLater {
            if (stage != null && stage.isShowing()) {
                stage.toFront()
                return
            }
            def panel = new YoloALPanel(qupath)
            stage = new Stage()
            stage.setTitle("YOLO Active Learning")
            stage.setScene(new Scene(panel.build(), 1100, 700))
            stage.initOwner(qupath.getStage())
            stage.setOnCloseRequest { panel.shutdown() }
            stage.show()
        }
    }
}
