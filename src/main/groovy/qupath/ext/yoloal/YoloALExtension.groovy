package qupath.ext.yoloal

import javafx.scene.control.MenuItem
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.extensions.QuPathExtension

/**
 * QuPath 0.6 extension entry point.
 * Registers "Extensions > YOLO Active Learning > Open Panel" menu item.
 */
class YoloALExtension implements QuPathExtension {

    @Override
    String getName() { "YOLO Active Learning" }

    @Override
    String getDescription() { "Train & run YOLO detectors with active learning inside QuPath" }

    @Override
    void installExtension(QuPathGUI qupath) {
        def menu = qupath.getMenu("Extensions>YOLO Active Learning", true)

        def openItem = new MenuItem("Open Panel…")
        openItem.setOnAction { new YoloALCommand(qupath).show() }
        menu.getItems().add(openItem)
    }
}
