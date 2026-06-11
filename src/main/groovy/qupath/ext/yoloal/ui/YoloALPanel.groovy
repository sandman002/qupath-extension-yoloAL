package qupath.ext.yoloal.ui

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import javafx.application.Platform
import javafx.collections.FXCollections
import javafx.collections.ObservableList
import javafx.geometry.Insets
import javafx.geometry.Orientation
import javafx.geometry.Pos
import javafx.scene.Node
import javafx.scene.chart.BarChart
import javafx.scene.chart.CategoryAxis
import javafx.scene.chart.LineChart
import javafx.scene.chart.NumberAxis
import javafx.scene.chart.XYChart
import javafx.beans.property.ReadOnlyStringWrapper
import javafx.beans.property.SimpleStringProperty
import javafx.scene.control.*
import javafx.scene.control.cell.ComboBoxTableCell
import javafx.scene.layout.*
import javafx.scene.text.Font
import javafx.stage.DirectoryChooser
import qupath.ext.yoloal.PatchExtractor
import qupath.ext.yoloal.ServerClient
import qupath.lib.gui.QuPathGUI
import qupath.lib.images.ImageData
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.ROIs

import java.awt.image.BufferedImage
import java.nio.file.Files
import java.nio.file.Paths
import java.text.SimpleDateFormat
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.logging.Logger

/**
 * Main JavaFX panel.
 *
 * Directory structure (all relative to the QuPath project directory):
 *   <project_dir>/<rootName>/
 *     data/   dataset_<date>_<px>px_<aug>aug/   ← one per extraction run
 *       images/{train,val,test}/
 *       labels/{train,val,test}/
 *       origins/{train,val,test}/               ← patch origin JSONs
 *       dataset.yaml   extract_log.json
 *     models/  train_<N>_<date>_<model>/        ← one per training run
 *       weights/best.pt   train_log.json
 *     inference/  infer_<N>_<date>/             ← one per inference run
 *       <SlideName>.geojson   infer_log.json
 */
class YoloALPanel {

    /** One row in the slide-role assignment table. */
    static class SlideRoleEntry {
        final String name
        final def    entry   // ProjectImageEntry
        final javafx.beans.property.StringProperty role
        final javafx.beans.property.StringProperty status = new SimpleStringProperty("new")

        SlideRoleEntry(String name, def entry, String initialRole) {
            this.name  = name
            this.entry = entry
            this.role  = new SimpleStringProperty(initialRole ?: "skip")
        }
    }

    static final Logger log  = Logger.getLogger(YoloALPanel.class.name)
    private final Gson   gson = new Gson()
    private final java.lang.reflect.Type mapType = new TypeToken<Map<String,Object>>(){}.getType()

    private static final List<String> NAME_ADJ  = [
        "brave","bright","calm","clever","dark","eager","fair","fancy","fierce","gentle",
        "grand","happy","jolly","keen","kind","lively","mighty","noble","odd","proud",
        "quick","rare","rich","safe","sharp","shy","slim","sly","smart","snappy",
        "solid","stark","stout","swift","tidy","tiny","vivid","warm","witty","young"
    ]
    private static final List<String> NAME_NOUN = [
        "ant","bear","bird","boar","buck","crow","deer","dove","duck","eagle",
        "elk","falcon","fawn","fox","frog","hawk","hare","heron","ibis","jay",
        "kite","lamb","lark","lion","lynx","mink","mole","moth","mule","newt",
        "owl","panda","pike","puma","quail","ram","raven","robin","rook","seal",
        "slug","snail","sparrow","stag","swan","toad","vole","wasp","wolf","wren"
    ]

    private final QuPathGUI qupath
    private final ServerClient client = new ServerClient()
    private Process serverProcess

    // ── top bar ───────────────────────────────────────────────────────────
    private TextField rootDirField = new TextField()   // full path to the yolo-al root
    private Label     statusLabel  = new Label("Server: not started")
    private Label     deviceLabel  = new Label("Device: —")
    private TextArea  serverLog    = new TextArea()

    // ── dataset / extract controls ────────────────────────────────────────
    private Spinner<Integer> patchSizeSpinner   = new Spinner<>(128, 2048, 640, 64)
    private Spinner<Integer> maxOffsetSpinner   = new Spinner<>(0, 512, 100, 10)
    private Spinner<Integer> augPerBoxSpinner   = new Spinner<>(1, 20, 1)
    private Spinner<Double>  valFractionSpinner = new Spinner<>(0.0, 0.5, 0.15, 0.05)
    private CheckBox         autoValCheck       = new CheckBox("Auto-split train → val")
    private Label      extractProgress  = new Label("")
    private Button     stopExtractBtn   = new Button("■ Stop")
    private final java.util.concurrent.atomic.AtomicBoolean extractCancelled = new java.util.concurrent.atomic.AtomicBoolean(false)
    private Label      activeDatasetLabel = new Label("No dataset extracted yet")
    private Button     openDatasetBtn     = new Button("📂 Open Folder")

    // ── slide role assignment table ───────────────────────────────────────────
    private ObservableList<SlideRoleEntry> slideRoleEntries = FXCollections.observableArrayList()
    private TableView<SlideRoleEntry>      slideTable       = new TableView<>(slideRoleEntries)
    private RadioButton  freshRadio        = new RadioButton("New dataset")
    private RadioButton  appendRadio       = new RadioButton("Add to existing")
    private ToggleGroup  extractModeGroup  = new ToggleGroup()
    private ComboBox<String> targetDatasetBox = new ComboBox<>()
    private Label        datasetInfoLabel  = new Label()

    // ── train controls ────────────────────────────────────────────────────
    private ComboBox<String>         datasetSelectBox = new ComboBox<>()
    private BarChart<String, Number> splitChart       = null
    private BarChart<String, Number> scaleChart       = null
    private ComboBox<String> modelBox      = new ComboBox<>(FXCollections.observableArrayList(
            "yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt",
            "yolo11n.pt","yolo11s.pt","yolo11m.pt"))
    private Spinner<Integer> epochsSpinner = new Spinner<>(1, 500, 50)
    private Spinner<Integer> batchSpinner  = new Spinner<>(1, 64, 8)
    private Spinner<Integer> imgsSpinner   = new Spinner<>(320, 1280, 640, 32)
    private Spinner<Double>  lrSpinner     = new Spinner<>(0.0001, 0.1, 0.01, 0.001)
    private Button           trainBtn      = new Button("▶ Start Training")
    private Button           stopBtn       = new Button("■ Stop")
    private Label            trainStatus   = new Label("Idle")
    private Label            metricsLabel  = new Label()

    // ── augmentation (advanced) ───────────────────────────────────────────
    private Spinner<Double> augDegrees   = new Spinner<>(0.0, 180.0, 10.0,  1.0)
    private Spinner<Double> augTranslate = new Spinner<>(0.0,   1.0,  0.2, 0.05)
    private Spinner<Double> augScale     = new Spinner<>(0.0,   1.0,  0.2, 0.05)
    private Spinner<Double> augFlipud    = new Spinner<>(0.0,   1.0,  0.5, 0.05)
    private Spinner<Double> augFliplr    = new Spinner<>(0.0,   1.0,  0.5, 0.05)
    private Spinner<Double> augMosaic    = new Spinner<>(0.0,   1.0,  0.1, 0.05)
    private Spinner<Double> augHsvH      = new Spinner<>(0.0,   1.0, 0.05, 0.01)
    private Spinner<Double> augHsvS      = new Spinner<>(0.0,   1.0,  0.3, 0.05)
    private Spinner<Double> augHsvV      = new Spinner<>(0.0,   1.0, 0.25, 0.05)

    // ── infer controls ────────────────────────────────────────────────────
    private ComboBox<String> modelSelectBox    = new ComboBox<>()
    private Spinner<Double>  confSpinner       = new Spinner<>(0.01, 1.0, 0.25, 0.05)
    private Spinner<Double>  iouSpinner        = new Spinner<>(0.01, 1.0, 0.45, 0.05)
    private Spinner<Integer> tileSizeSpinner   = new Spinner<>(128, 2048, 640, 64)
    private Spinner<Integer> tileOverlapSpinner= new Spinner<>(0, 512, 64, 16)
    private Spinner<Double>  tissueFracSpinner = new Spinner<>(0.0, 1.0, 0.15, 0.05)
    private CheckBox         debugTilesChk     = new CheckBox("Save debug tiles")
    private CheckBox         scale1xChk        = new CheckBox("1×")
    private CheckBox         scale2xChk        = new CheckBox("2×")
    private CheckBox         scale4xChk        = new CheckBox("4×")
    private CheckBox         scale8xChk        = new CheckBox("8×")
    private Button           planBtn           = new Button("🔍 Plan")
    private Button           inferBtn          = new Button("▶ Run Inference")

    // ── log areas ─────────────────────────────────────────────────────────
    private TextArea extractLog = new TextArea()
    private TextArea inferLog   = new TextArea()

    // ── train charts ──────────────────────────────────────────────────────
    private XYChart.Series<Number,Number> trainLossSeries = new XYChart.Series<>(name: "Train loss")
    private XYChart.Series<Number,Number> valLossSeries   = new XYChart.Series<>(name: "Val loss")
    private XYChart.Series<Number,Number> map50Series     = new XYChart.Series<>(name: "mAP@50")

    // ── state ─────────────────────────────────────────────────────────────
    private ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor()
    private int    lastEpoch       = 0
    private String activeDatasetDir = null   // path set after successful extraction
    private String activeModelPath  = null   // path set after successful training

    YoloALPanel(QuPathGUI qupath) {
        this.qupath = qupath
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Root
    // ═══════════════════════════════════════════════════════════════════════

    Node build() {
        def root = new BorderPane()
        root.setPadding(new Insets(8))
        root.setTop(buildTopBar())

        def split = new SplitPane(buildLeftPanel(), buildRightPanel())
        split.setDividerPositions(0.38)
        root.setCenter(split)

        statusLabel.setFont(Font.font("Monospaced", 11))
        statusLabel.setPadding(new Insets(4, 0, 0, 0))
        root.setBottom(statusLabel)
        return root
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Top bar
    // ═══════════════════════════════════════════════════════════════════════

    private Node buildTopBar() {
        def serverBtn = new Button("▶ Start Server")
        serverBtn.setOnAction { startServer() }

        def flushBtn = new Button("⏹ Flush Server")
        flushBtn.setStyle("-fx-text-fill: #c00;")
        flushBtn.setOnAction { flushServer() }

        rootDirField.setPromptText("Output root dir  (e.g. yolo_sandman)")
        HBox.setHgrow(rootDirField, Priority.ALWAYS)

        // Auto button: set to <project_dir>/<typed-name or "yolo-al">
        def autoBtn = new Button("Auto")
        autoBtn.setTooltip(new Tooltip("Set to <QuPath project dir>/<folder name>"))
        autoBtn.setOnAction {
            def pd = projectDir()
            if (!pd) { alert("Open a QuPath project first."); return }
            def name = rootDirField.getText()?.trim()
            if (!name) name = "yolo-al"
            // If the field already has a full path keep it; if it looks like a name, make it a path
            if (!new File(name).isAbsolute()) {
                rootDirField.setText(new File(pd, name).absolutePath)
            }
        }

        // Browse button: pick any directory
        def browseBtn = new Button("…")
        browseBtn.setTooltip(new Tooltip("Browse for output root directory"))
        browseBtn.setOnAction {
            def dc = new DirectoryChooser()
            dc.setTitle("Select output root directory")
            def current = rootDirField.getText()?.trim()
            if (current) {
                def f = new File(current)
                if (f.exists()) dc.setInitialDirectory(f)
                else if (f.parentFile?.exists()) dc.setInitialDirectory(f.parentFile)
            }
            def dir = dc.showDialog(qupath.getStage())
            if (dir) rootDirField.setText(dir.absolutePath)
        }

        deviceLabel.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11; -fx-text-fill: #555;")

        def aboutBtn = new Button("ℹ")
        aboutBtn.setTooltip(new Tooltip("About YOLO Active Learning"))
        aboutBtn.setOnAction { showAbout() }

        def bar = new HBox(8,
            new Label("Output dir:"), rootDirField, autoBtn, browseBtn,
            new Separator(),
            serverBtn, flushBtn, deviceLabel,
            new Separator(), aboutBtn)
        bar.setAlignment(Pos.CENTER_LEFT)
        bar.setPadding(new Insets(0, 0, 6, 0))
        return bar
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Left panel
    // ═══════════════════════════════════════════════════════════════════════

    private Node buildLeftPanel() {
        def tabs = new TabPane()
        tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
        tabs.getTabs().addAll(
            new Tab("1 · Dataset", buildDatasetControls()),
            new Tab("2 · Train",   buildTrainControls()),
            new Tab("3 · Infer",   buildInferControls())
        )
        return tabs
    }

    private Node buildDatasetControls() {
        def grid = new GridPane()
        grid.setHgap(10); grid.setVgap(8); grid.setPadding(new Insets(14))

        [patchSizeSpinner, maxOffsetSpinner, augPerBoxSpinner, valFractionSpinner].each {
            it.setEditable(true); it.setMaxWidth(Double.MAX_VALUE)
            GridPane.setHgrow(it, Priority.ALWAYS)
        }

        int r = 0
        valFractionSpinner.setEditable(true); valFractionSpinner.setMaxWidth(Double.MAX_VALUE)
        valFractionSpinner.setDisable(true)
        autoValCheck.setSelected(false); autoValCheck.setStyle("-fx-font-size: 11;")
        autoValCheck.selectedProperty().addListener { obs, old, nw -> valFractionSpinner.setDisable(!nw) }

        grid.addRow(r++, new Label("Patch size (px):"),   patchSizeSpinner)
        grid.addRow(r++, new Label("Max random offset:"), maxOffsetSpinner)
        grid.addRow(r++, new Label("Augments per bbox:"), augPerBoxSpinner)
        grid.add(autoValCheck, 0, r, 2, 1); r++
        grid.addRow(r++, new Label("  Val fraction:"),    valFractionSpinner)
        grid.add(new Separator(), 0, r++, 2, 1)

        // ── Extraction mode ─────────────────────────────────────────────────
        freshRadio.setToggleGroup(extractModeGroup)
        appendRadio.setToggleGroup(extractModeGroup)
        if (extractModeGroup.getSelectedToggle() == null) freshRadio.setSelected(true)
        targetDatasetBox.setMaxWidth(Double.MAX_VALUE)
        targetDatasetBox.setPromptText("Select dataset to add to…")
        targetDatasetBox.setDisable(true)
        def dsPickRefreshBtn = new Button("⟳")
        dsPickRefreshBtn.setTooltip(new Tooltip("Scan for existing datasets"))
        dsPickRefreshBtn.setOnAction { populateTargetDatasetBox() }
        def dsPickRow = new HBox(4, targetDatasetBox, dsPickRefreshBtn)
        HBox.setHgrow(targetDatasetBox, Priority.ALWAYS)
        dsPickRefreshBtn.setDisable(true)

        appendRadio.selectedProperty().addListener { obs, old, nw ->
            targetDatasetBox.setDisable(!nw)
            dsPickRefreshBtn.setDisable(!nw)
            if (nw) { populateTargetDatasetBox(); updateDatasetInfoLabel() }
            else    { datasetInfoLabel.setText("") }
        }
        def modeBox = new HBox(10, freshRadio, appendRadio)
        modeBox.setAlignment(Pos.CENTER_LEFT)
        grid.add(new Label("Mode:"), 0, r); grid.add(modeBox, 1, r++)
        grid.add(new Label("Target:"), 0, r); grid.add(dsPickRow, 1, r++)

        datasetInfoLabel.setStyle("-fx-font-size: 10; -fx-text-fill: #0a7; -fx-font-family: Monospaced;")
        datasetInfoLabel.setWrapText(true)
        grid.add(datasetInfoLabel, 0, r++, 2, 1)
        grid.add(new Separator(), 0, r++, 2, 1)

        // ── Slide role assignment ────────────────────────────────────────────
        def scanBtn = new Button("⟳ Scan Slides")
        scanBtn.setTooltip(new Tooltip("Load all project slides and their current roles"))
        scanBtn.setOnAction { scanSlides() }

        def metaBtn = new Button("From Metadata")
        metaBtn.setTooltip(new Tooltip("Fill roles from project metadata (role=train/val/test/skip)"))
        metaBtn.setOnAction { populateRolesFromMetadata() }

        def allTrainBtn = new Button("All → train")
        allTrainBtn.setOnAction {
            slideRoleEntries.each { it.role.set("train") }
            slideTable.refresh()
            runBg { slideRoleEntries.each { saveRoleToMetadata(it) } }
        }

        def tableBar = new HBox(5, scanBtn, metaBtn, allTrainBtn)
        grid.add(new Label("Slides:"), 0, r); grid.add(tableBar, 1, r++)

        // Table columns (build once)
        if (slideTable.getColumns().isEmpty()) {
            slideTable.setEditable(true)
            slideTable.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY)
            slideTable.setPlaceholder(new Label("Click ⟳ Scan Slides to load project slides"))

            def nameCol = new TableColumn<SlideRoleEntry, String>("Slide")
            nameCol.setCellValueFactory { f -> new ReadOnlyStringWrapper(f.value.name) }

            def roleCol = new TableColumn<SlideRoleEntry, String>("Role")
            roleCol.setCellValueFactory { f -> f.value.role }
            roleCol.setCellFactory(ComboBoxTableCell.forTableColumn(
                FXCollections.observableArrayList("train", "val", "test", "skip")))
            roleCol.setOnEditCommit { e ->
                e.rowValue.role.set(e.newValue)
                runBg { saveRoleToMetadata(e.rowValue) }
            }
            roleCol.setPrefWidth(72); roleCol.setMinWidth(72); roleCol.setMaxWidth(80)

            def statusCol = new TableColumn<SlideRoleEntry, String>("Status")
            statusCol.setCellValueFactory { f -> f.value.status }
            statusCol.setCellFactory(ComboBoxTableCell.forTableColumn(
                FXCollections.observableArrayList("new", "done", "redo")))
            statusCol.setOnEditCommit { e -> e.rowValue.status.set(e.newValue) }
            statusCol.setPrefWidth(60); statusCol.setMinWidth(55); statusCol.setMaxWidth(70)

            slideTable.getColumns().addAll(nameCol, roleCol, statusCol)
        }
        slideTable.setPrefHeight(175)
        grid.add(slideTable, 0, r++, 2, 1)

        def hint = new Label(
            "Role set in the table overrides project metadata.\n" +
            "Status: new = not yet extracted  |  done = already in dataset.")
        hint.setWrapText(true)
        hint.setStyle("-fx-text-fill: #666; -fx-font-size: 10;")
        grid.add(hint, 0, r++, 2, 1)

        activeDatasetLabel.setStyle("-fx-font-family: Monospaced; -fx-font-size: 10; -fx-text-fill: #0a7;")
        activeDatasetLabel.setWrapText(true)
        openDatasetBtn.setDisable(true)
        openDatasetBtn.setTooltip(new Tooltip("Open extracted dataset folder in Explorer"))
        openDatasetBtn.setOnAction {
            if (activeDatasetDir) {
                try { java.awt.Desktop.getDesktop().open(new File(activeDatasetDir)) }
                catch (Exception e) { alert("Could not open folder:\n${e.message}") }
            }
        }
        def activeDsRow = new HBox(6, activeDatasetLabel, openDatasetBtn)
        HBox.setHgrow(activeDatasetLabel, Priority.ALWAYS)
        activeDsRow.setAlignment(Pos.CENTER_LEFT)
        grid.add(activeDsRow, 0, r++, 2, 1)

        def extractBtn = new Button("Extract Patches")
        extractBtn.setMaxWidth(Double.MAX_VALUE)
        extractBtn.setOnAction { extractAll() }

        stopExtractBtn.setDisable(true)
        stopExtractBtn.setOnAction {
            extractCancelled.set(true); stopExtractBtn.setDisable(true)
            appendExtractLog("Stopping after current slide…\n")
        }

        extractProgress.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11; -fx-text-fill: #555;")
        extractProgress.setWrapText(true)

        def btnRow = new HBox(6, extractBtn, stopExtractBtn)
        HBox.setHgrow(extractBtn, Priority.ALWAYS)
        grid.add(btnRow,          0, r++, 2, 1)
        grid.add(extractProgress, 0, r++, 2, 1)

        def c1 = new ColumnConstraints(); c1.setHgrow(Priority.NEVER)
        def c2 = new ColumnConstraints(); c2.setHgrow(Priority.ALWAYS)
        grid.getColumnConstraints().addAll(c1, c2)
        return new ScrollPane(grid).tap { fitToWidth = true; hbarPolicy = ScrollPane.ScrollBarPolicy.NEVER }
    }

    private Node buildTrainControls() {
        def grid = new GridPane()
        grid.setHgap(10); grid.setVgap(10); grid.setPadding(new Insets(14))

        def dconv = new javafx.util.converter.DoubleStringConverter()
        [epochsSpinner, batchSpinner, imgsSpinner, lrSpinner].each { it.setEditable(true); it.setMaxWidth(Double.MAX_VALUE) }
        modelBox.setMaxWidth(Double.MAX_VALUE)
        lrSpinner.getValueFactory().setConverter(dconv)

        datasetSelectBox.setMaxWidth(Double.MAX_VALUE)
        datasetSelectBox.setPromptText("Select dataset…")
        def dsRefreshBtn = new Button("⟳")
        dsRefreshBtn.setTooltip(new Tooltip("Scan for extracted datasets"))
        dsRefreshBtn.setOnAction { refreshDatasetList() }
        def dsRow = new HBox(4, datasetSelectBox, dsRefreshBtn)
        HBox.setHgrow(datasetSelectBox, Priority.ALWAYS)
        datasetSelectBox.valueProperty().addListener { obs, old, nw -> if (nw) refreshSplitChart(nw) }

        int r = 0
        grid.add(new Label("Dataset:"), 0, r); grid.add(dsRow, 1, r++)
        grid.addRow(r++, new Label("Model:"),      modelBox)
        grid.addRow(r++, new Label("Epochs:"),     epochsSpinner)
        grid.addRow(r++, new Label("Batch:"),      batchSpinner)
        grid.addRow(r++, new Label("Image size:"), imgsSpinner)
        grid.addRow(r++, new Label("LR0:"),        lrSpinner)

        stopBtn.setDisable(true)
        trainBtn.setMaxWidth(Double.MAX_VALUE); stopBtn.setMaxWidth(Double.MAX_VALUE)
        trainBtn.setOnAction { startTraining() }; stopBtn.setOnAction { stopTraining() }

        def btnRow = new HBox(8, trainBtn, stopBtn)
        HBox.setHgrow(trainBtn, Priority.ALWAYS); HBox.setHgrow(stopBtn, Priority.ALWAYS)

        trainStatus.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11;")
        metricsLabel.setWrapText(true)
        metricsLabel.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11; -fx-text-fill: #0a7;")

        grid.add(btnRow,       0, r++, 2, 1)
        grid.add(trainStatus,  0, r++, 2, 1)
        grid.add(metricsLabel, 0, r++, 2, 1)

        def catAxis = new CategoryAxis()
        def cntAxis = new NumberAxis()
        cntAxis.setLabel("Patches")
        splitChart = new BarChart<>(catAxis, cntAxis)
        splitChart.setTitle("Dataset splits")
        splitChart.setAnimated(false)
        splitChart.setLegendVisible(false)
        splitChart.setPrefHeight(160)
        splitChart.setMaxHeight(200)
        grid.add(splitChart, 0, r++, 2, 1)

        def scatAxis = new CategoryAxis()
        def scntAxis = new NumberAxis()
        scntAxis.setLabel("Patches")
        scaleChart = new BarChart<>(scatAxis, scntAxis)
        scaleChart.setTitle("Patches per scale")
        scaleChart.setAnimated(false)
        scaleChart.setLegendVisible(false)
        scaleChart.setPrefHeight(150)
        scaleChart.setMaxHeight(180)
        grid.add(scaleChart, 0, r++, 2, 1)

        // ── Advanced Augmentation ─────────────────────────────────────────
        [augDegrees, augTranslate, augScale, augFlipud, augFliplr,
         augMosaic, augHsvH, augHsvS, augHsvV].each {
            it.setEditable(true); it.setMaxWidth(Double.MAX_VALUE)
            it.getValueFactory().setConverter(dconv)
        }
        def ag = new GridPane(); ag.setHgap(10); ag.setVgap(6); ag.setPadding(new Insets(8))
        int ar = 0
        ag.addRow(ar++, new Label("Rotation (°):"),     augDegrees)
        ag.addRow(ar++, new Label("Translate:"),         augTranslate)
        ag.addRow(ar++, new Label("Scale:"),             augScale)
        ag.addRow(ar++, new Label("Flip U-D prob:"),     augFlipud)
        ag.addRow(ar++, new Label("Flip L-R prob:"),     augFliplr)
        ag.addRow(ar++, new Label("Mosaic prob:"),       augMosaic)
        ag.addRow(ar++, new Label("HSV Hue:"),           augHsvH)
        ag.addRow(ar++, new Label("HSV Saturation:"),    augHsvS)
        ag.addRow(ar++, new Label("HSV Value:"),         augHsvV)
        def ac1 = new ColumnConstraints(); ac1.setHgrow(Priority.NEVER)
        def ac2 = new ColumnConstraints(); ac2.setHgrow(Priority.ALWAYS)
        ag.getColumnConstraints().addAll(ac1, ac2)

        def augPane = new TitledPane("Advanced Augmentation", ag)
        augPane.setCollapsible(true); augPane.setExpanded(false)
        grid.add(augPane, 0, r++, 2, 1)

        def c1 = new ColumnConstraints(); c1.setHgrow(Priority.NEVER)
        def c2 = new ColumnConstraints(); c2.setHgrow(Priority.ALWAYS)
        grid.getColumnConstraints().addAll(c1, c2)
        return new ScrollPane(grid).tap { fitToWidth = true; hbarPolicy = ScrollPane.ScrollBarPolicy.NEVER }
    }

    private Node buildInferControls() {
        def grid = new GridPane()
        grid.setHgap(10); grid.setVgap(10); grid.setPadding(new Insets(14))

        def dconv = new javafx.util.converter.DoubleStringConverter()
        [confSpinner, iouSpinner, tissueFracSpinner].each {
            it.setEditable(true); it.setMaxWidth(Double.MAX_VALUE)
            it.getValueFactory().setConverter(dconv)
        }
        [tileSizeSpinner, tileOverlapSpinner].each { it.setEditable(true); it.setMaxWidth(Double.MAX_VALUE) }
        modelSelectBox.setMaxWidth(Double.MAX_VALUE)
        modelSelectBox.setPromptText("Select trained model…")

        def refreshBtn = new Button("⟳")
        refreshBtn.setTooltip(new Tooltip("Scan for trained models"))
        refreshBtn.setOnAction { refreshModelList() }

        // Auto-populate scale checkboxes when a model is selected
        scale1xChk.setSelected(true)
        [scale1xChk, scale2xChk, scale4xChk, scale8xChk].each {
            it.setTooltip(new Tooltip("Include this downsample scale in the inference scan"))
        }
        modelSelectBox.valueProperty().addListener { obs, old, nw ->
            if (nw) applyModelScales(nw)
        }

        int r = 0
        def modelRow = new HBox(4, modelSelectBox, refreshBtn)
        HBox.setHgrow(modelSelectBox, Priority.ALWAYS)
        grid.add(new Label("Model:"), 0, r); grid.add(modelRow, 1, r++)
        grid.addRow(r++, new Label("Confidence:"),    confSpinner)
        grid.addRow(r++, new Label("NMS IoU:"),       iouSpinner)

        def sep = new Separator(); grid.add(sep, 0, r++, 2, 1)

        grid.addRow(r++, new Label("Tile size (px):"),   tileSizeSpinner)
        grid.addRow(r++, new Label("Tile overlap (px):"),tileOverlapSpinner)
        grid.addRow(r++, new Label("Tissue threshold:"), tissueFracSpinner)

        def scaleBox = new HBox(10, new Label("Infer scales:"),
            scale1xChk, scale2xChk, scale4xChk, scale8xChk)
        scaleBox.setAlignment(Pos.CENTER_LEFT)
        Tooltip.install(scaleBox, new Tooltip(
            "Downsample scales to scan.\nMatch the scales used during patch extraction.\n" +
            "Auto-populated when a model with training metadata is selected."))
        grid.add(scaleBox, 0, r++, 2, 1)

        debugTilesChk.setTooltip(new Tooltip("Save every inference tile as a JPEG to output_dir/debug_tiles/\nUse to verify color space and tile content. Disable for production."))
        grid.add(debugTilesChk, 1, r++)

        def roiHint = new Label(
            "ROI mode (auto-detected):\n" +
            "  • If annotations exist on the slide → tiles only inside those ROIs\n" +
            "  • Otherwise → tissue is detected at 1/32 resolution;\n" +
            "    tiles are generated within the tissue bounding box\n\n" +
            "Results → <root>/inference/infer_N_date/\n" +
            "  <slide>.geojson  (drag-drop onto QuPath viewer)\n" +
            "  infer_log.json   (model, conf, counts)\n\n" +
            "Green = confident ≥ conf thresh\n" +
            "Red   = uncertain < conf thresh\n\n" +
            "Active-learning loop:\n" +
            "  Accept → keep  |  Reject → Delete\n" +
            "  New → draw + classify  →  re-train")
        roiHint.setWrapText(true)
        roiHint.setStyle("-fx-text-fill: #555; -fx-font-size: 11;")
        grid.add(roiHint, 0, r++, 2, 1)

        planBtn.setMaxWidth(Double.MAX_VALUE)
        planBtn.setStyle("-fx-text-fill: #00a;")
        planBtn.setOnAction { planInference() }
        inferBtn.setMaxWidth(Double.MAX_VALUE)
        inferBtn.setOnAction { runInference() }

        def btnRow = new HBox(6, planBtn, inferBtn)
        HBox.setHgrow(inferBtn, Priority.ALWAYS)
        grid.add(btnRow, 0, r++, 2, 1)

        def c1 = new ColumnConstraints(); c1.setHgrow(Priority.NEVER)
        def c2 = new ColumnConstraints(); c2.setHgrow(Priority.ALWAYS)
        grid.getColumnConstraints().addAll(c1, c2)
        return new ScrollPane(grid).tap { fitToWidth = true; hbarPolicy = ScrollPane.ScrollBarPolicy.NEVER }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Right panel

    private Node buildRightPanel() {
        serverLog.setEditable(false); serverLog.setWrapText(false)
        serverLog.setStyle(
            "-fx-font-family: 'Consolas','Courier New',Monospaced; -fx-font-size: 11; " +
            "-fx-control-inner-background: #1e1e1e; -fx-text-fill: #d4d4d4;")
        serverLog.setPromptText("Python server output…")
        VBox.setVgrow(serverLog, Priority.ALWAYS)

        def serverSection = new VBox(4, labeledHeader("Server Log"), serverLog)
        VBox.setVgrow(serverSection, Priority.ALWAYS)

        extractLog.setEditable(false)
        extractLog.setStyle("-fx-font-family: 'Consolas','Courier New',Monospaced; -fx-font-size: 11;")
        VBox.setVgrow(extractLog, Priority.ALWAYS)

        inferLog.setEditable(false)
        inferLog.setStyle("-fx-font-family: 'Consolas','Courier New',Monospaced; -fx-font-size: 11;")
        VBox.setVgrow(inferLog, Priority.ALWAYS)

        def taskTabs = new TabPane()
        taskTabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
        taskTabs.getTabs().addAll(
            new Tab("Extract Log",  fillVBox(extractLog)),
            new Tab("Train Charts", buildTrainCharts()),
            new Tab("Infer Log",    fillVBox(inferLog))
        )

        def vSplit = new SplitPane()
        vSplit.setOrientation(Orientation.VERTICAL)
        vSplit.getItems().addAll(serverSection, taskTabs)
        vSplit.setDividerPositions(0.35)
        return vSplit
    }

    private Node buildTrainCharts() {
        def xA1 = new NumberAxis(); xA1.setLabel("Epoch")
        def yA1 = new NumberAxis(); yA1.setLabel("Loss")
        def lossChart = new LineChart<>(xA1, yA1)
        lossChart.setTitle("Training Loss")
        lossChart.getData().addAll(trainLossSeries, valLossSeries)
        lossChart.setAnimated(false); VBox.setVgrow(lossChart, Priority.ALWAYS)

        def xA2 = new NumberAxis(); xA2.setLabel("Epoch")
        def yA2 = new NumberAxis(); yA2.setLabel("mAP")
        def mapChart = new LineChart<>(xA2, yA2)
        mapChart.setTitle("mAP@50")
        mapChart.getData().add(map50Series)
        mapChart.setAnimated(false); VBox.setVgrow(mapChart, Priority.ALWAYS)

        def box = new VBox(8, lossChart, mapChart)
        box.setPadding(new Insets(6)); VBox.setVgrow(box, Priority.ALWAYS)
        return box
    }

    private static Node fillVBox(TextArea a) {
        def b = new VBox(a); VBox.setVgrow(a, Priority.ALWAYS); b.setPadding(new Insets(4)); return b
    }

    private static Label labeledHeader(String t) {
        def l = new Label(" $t "); l.setStyle("-fx-font-weight: bold; -fx-font-size: 11; -fx-text-fill: #888;"); return l
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Directory helpers
    // ═══════════════════════════════════════════════════════════════════════

    private File projectDir() {
        try {
            def p = qupath.getProject()?.getPath()?.getParent()?.toFile()
            return p
        } catch (Exception ignored) { return null }
    }

    private File yoloAlRoot() {
        def path = rootDirField.getText()?.trim()
        if (!path) {
            // Auto-derive from project dir
            def pd = projectDir()
            path = pd ? new File(pd, "yolo-al").absolutePath
                      : new File(System.getProperty("user.home"), "yolo-al").absolutePath
            Platform.runLater { rootDirField.setText(path) }
        }
        def root = new File(path)
        root.mkdirs()
        return root
    }

    private String todayStr() {
        new SimpleDateFormat("yyyy-MM-dd").format(new Date())
    }

    /** Creates and returns next data cycle directory with a memorable random name. */
    private File nextDataDir() {
        def base = new File(yoloAlRoot(), "data")
        base.mkdirs()
        def rng  = new Random()
        def adj  = NAME_ADJ [rng.nextInt(NAME_ADJ.size())]
        def noun = NAME_NOUN[rng.nextInt(NAME_NOUN.size())]
        def name = "${adj}_${noun}_${todayStr()}"
        def dir  = new File(base, name)
        // Avoid collision (extremely rare) by appending a counter
        int n = 2
        while (dir.exists()) { dir = new File(base, "${name}_${n++}") }
        dir.mkdirs()
        return dir
    }

    /** Returns the most-recently-modified data directory, or null. */
    private File latestDataDir() {
        def base = new File(yoloAlRoot(), "data")
        if (!base.exists()) return null
        return base.listFiles({ f -> f.isDirectory() } as FileFilter)
                   ?.sort { it.lastModified() }?.last()
    }

    /** Creates next training output directory and returns its path. */
    private File nextModelDir(String modelName) {
        def base = new File(yoloAlRoot(), "models")
        base.mkdirs()
        def rng  = new Random()
        def adj  = NAME_ADJ [rng.nextInt(NAME_ADJ.size())]
        def noun = NAME_NOUN[rng.nextInt(NAME_NOUN.size())]
        def name = "train_${adj}_${noun}_${todayStr()}"
        def dir  = new File(base, name)
        int n = 2
        while (dir.exists()) { dir = new File(base, "${name}_${n++}") }
        dir.mkdirs()
        return dir
    }

    /** Scans data/ for extracted dataset directories, updates the dataset ComboBox. */
    private void refreshDatasetList() {
        def base = new File(yoloAlRoot(), "data")
        if (!base.exists()) return
        def datasets = base.listFiles({ f -> f.isDirectory() && new File(f, "dataset.yaml").exists() } as FileFilter)
            ?.sort { it.lastModified() }
            ?.collect { it.absolutePath }
            ?: []
        Platform.runLater {
            datasetSelectBox.setItems(FXCollections.observableArrayList(datasets))
            // Prefer activeDatasetDir if set, else latest
            def preferred = activeDatasetDir ?: (datasets.isEmpty() ? null : datasets.last())
            if (preferred && datasets.contains(preferred)) datasetSelectBox.setValue(preferred)
            else if (!datasets.isEmpty()) datasetSelectBox.setValue(datasets.last())
        }
    }

    /** Reads extract_log.json and updates the split + scale bar charts in the Train tab. */
    private void refreshSplitChart(String dataDir) {
        def logFile = new File(dataDir, "extract_log.json")
        if (!logFile.exists()) return
        try {
            def logData   = gson.fromJson(logFile.text, mapType)
            def slides    = logData.get("slides") as List ?: []
            def counts    = [train: 0, val: 0, test: 0]
            slides.each { s ->
                String role = (s as Map).get("role") as String ?: ""
                int patches = ((s as Map).get("patches") as Number)?.intValue() ?: 0
                if (counts.containsKey(role)) counts[role] += patches
            }
            Map scaleStats = logData.get("scaleStats") as Map ?: [:]
            Platform.runLater {
                if (splitChart != null) {
                    splitChart.getData().clear()
                    def series = new XYChart.Series<String, Number>()
                    series.getData().add(new XYChart.Data<>("train", counts.train))
                    series.getData().add(new XYChart.Data<>("val",   counts.val))
                    series.getData().add(new XYChart.Data<>("test",  counts.test))
                    splitChart.getData().add(series)
                }
                if (scaleChart != null) {
                    scaleChart.getData().clear()
                    def ss = new XYChart.Series<String, Number>()
                    ["1×", "2×", "4×", "8×"].each { k ->
                        int cnt = ((scaleStats.get(k) as Number)?.intValue() ?: 0)
                        if (cnt > 0) ss.getData().add(new XYChart.Data<>(k, cnt))
                    }
                    if (!ss.getData().isEmpty()) scaleChart.getData().add(ss)
                }
            }
        } catch (Exception ignored) {}
    }

    /** Scans models/ for best.pt files, updates the model ComboBox. */
    private void refreshModelList() {
        def base = new File(yoloAlRoot(), "models")
        if (!base.exists()) return
        def models = base.listFiles({ f -> f.isDirectory() } as FileFilter)
            ?.collect { new File(it, "weights/best.pt") }
            ?.findAll { it.exists() }
            ?.sort { it.lastModified() }
            ?.collect { it.absolutePath }
            ?: []
        Platform.runLater {
            modelSelectBox.setItems(FXCollections.observableArrayList(models))
            if (!models.isEmpty()) modelSelectBox.setValue(models.last())
        }
    }

    /** Creates next inference output directory. */
    private File nextInferDir() {
        def base = new File(yoloAlRoot(), "inference")
        base.mkdirs()
        int n = (base.listFiles({ f -> f.isDirectory() } as FileFilter)?.size() ?: 0) + 1
        def dir = new File(base, String.format("infer_%02d_%s", n, todayStr()))
        dir.mkdirs()
        return dir
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Actions — server
    // ═══════════════════════════════════════════════════════════════════════

    private void startServer() {
        serverLog.clear()
        Platform.runLater { deviceLabel.setText("Device: …") }
        setStatus("Connecting…")

        runBg {
            if (client.isAlive()) {
                Platform.runLater { serverLog.appendText("[INFO] Server already running.\n") }
                showServerReady()
                return
            }
            def pythonExe  = locatePython()
            def scriptPath = locateServerScript()
            setStatus("Starting Python server…")
            serverProcess = client.startServer(pythonExe, scriptPath, 5005, yoloAlRoot())

            def reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(serverProcess.getInputStream()))
            Thread.start {
                try { String l; while ((l = reader.readLine()) != null) {
                    final String ll = l; Platform.runLater { serverLog.appendText(ll + "\n") }
                }} catch (Exception ignored) {}
            }

            int tries = 0
            while (!client.isAlive() && tries++ < 40) Thread.sleep(500)
            if (client.isAlive()) showServerReady()
            else Platform.runLater { setStatus("Server: FAILED — see server log →"); deviceLabel.setText("Device: —") }
        }
    }

    private void flushServer() {
        setStatus("Stopping server…")
        runBg {
            try { client.shutdown() } catch (Exception ignored) {}
            serverProcess?.destroyForcibly(); serverProcess = null
            killProcessOnPort(5005)
            Platform.runLater {
                setStatus("Server: stopped"); deviceLabel.setText("Device: —")
                deviceLabel.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11; -fx-text-fill: #555;")
                serverLog.appendText("[INFO] Server stopped.\n")
            }
        }
    }

    /**
     * Read the extract_log.json for the dataset that produced this model and
     * tick the scale checkboxes to match training.
     */
    private void applyModelScales(String modelPath) {
        try {
            def modelDir  = new File(modelPath).parentFile?.parentFile  // weights/ → train_xxx/
            def trainLog  = new File(modelDir, "train_log.json")
            if (!trainLog.exists()) return
            def tl        = gson.fromJson(trainLog.text, mapType) as Map
            String dsDir  = tl?.get("dataset") as String
            if (!dsDir) return
            def extractLog = new File(dsDir, "extract_log.json")
            if (!extractLog.exists()) return
            def el        = gson.fromJson(extractLog.text, mapType) as Map
            List dsList   = el?.get("downsamples") as List ?: [1.0]
            Set<Double> ds = dsList.collect { (it as Number).doubleValue() } as Set
            Platform.runLater {
                scale1xChk.setSelected(ds.contains(1.0))
                scale2xChk.setSelected(ds.contains(2.0))
                scale4xChk.setSelected(ds.contains(4.0))
                scale8xChk.setSelected(ds.contains(8.0))
            }
        } catch (Exception ignored) {}
    }

    /** Returns the list of downsample values currently checked by the user. */
    private List<Double> selectedDownsamples() {
        def ds = []
        if (scale1xChk.isSelected()) ds << 1.0
        if (scale2xChk.isSelected()) ds << 2.0
        if (scale4xChk.isSelected()) ds << 4.0
        if (scale8xChk.isSelected()) ds << 8.0
        return ds ?: [1.0]  // always at least 1×
    }

    private void killProcessOnPort(int port) {
        try {
            def find = new ProcessBuilder("cmd", "/c",
                    "netstat -ano | findstr LISTENING | findstr :${port}")
                .redirectErrorStream(true).start()
            def out = find.inputStream.text.trim()
            find.waitFor()
            out.split("\n").each { line ->
                def parts = line.trim().split("\\s+")
                if (parts.length >= 5) {
                    def pid = parts[-1].trim()
                    if (pid.matches("\\d+") && pid != "0") {
                        log.info("Killing PID ${pid} on port ${port}")
                        new ProcessBuilder("taskkill", "/F", "/PID", pid)
                            .redirectErrorStream(true).start().waitFor()
                    }
                }
            }
        } catch (Exception ignored) {}
    }

    private void showServerReady() {
        def dev = client.getDevice()
        Platform.runLater {
            setStatus("Server: running at ${client.baseUrl}")
            deviceLabel.setText("Device: ${dev}")
            deviceLabel.setStyle(dev.contains("GPU")
                ? "-fx-font-family: Monospaced; -fx-font-size: 11; -fx-text-fill: #0a7; -fx-font-weight: bold;"
                : "-fx-font-family: Monospaced; -fx-font-size: 11; -fx-text-fill: #a70; -fx-font-weight: bold;")
            refreshDatasetList()
            refreshModelList()
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Actions — extract
    // ═══════════════════════════════════════════════════════════════════════

    private void extractAll() {
        def project = qupath.getProject()
        if (!project) { alert("Open a QuPath project first."); return }

        boolean continueMode = appendRadio.isSelected()
        if (continueMode && !targetDatasetBox.getValue()) {
            alert("Select a target dataset first (click ⟳ next to Target)."); return
        }
        def dataDir = continueMode ? new File(targetDatasetBox.getValue()) : nextDataDir()

        // Role map from table (takes priority over metadata when populated)
        Map<String, String> tableRoles = slideRoleEntries
            .collectEntries { [(it.name): it.role.get()] }

        // In continue mode: load existing log to know which slides are done
        Set<String> alreadyDone = continueMode ? extractedSlideNames(dataDir as File) : [] as Set
        // Slides explicitly marked "redo" in the table bypass the already-done check
        Set<String> redoSlides  = slideRoleEntries.findAll { it.status.get() == "redo" }.collect { it.name } as Set
        Map existingLog = [:]
        if (continueMode) {
            def lf = new File(dataDir, "extract_log.json")
            if (lf.exists()) { try { existingLog = gson.fromJson(lf.text, mapType) as Map } catch (Exception ignored) {} }
        }

        extractLog.clear()
        extractCancelled.set(false)
        stopExtractBtn.setDisable(false)
        Platform.runLater { extractProgress.setText("") }

        appendExtractLog("${continueMode ? 'Adding to' : 'Creating'}: ${dataDir.absolutePath}\n\n")

        runBg {
            def extractor = new PatchExtractor(
                patchSize:   patchSizeSpinner.getValue(),
                maxOffset:   maxOffsetSpinner.getValue(),
                augPerBox:   augPerBoxSpinner.getValue(),
                datasetRoot: dataDir.absolutePath,
                cancelFlag:  extractCancelled
            )

            // Seed totals from existing log when continuing
            def allClasses = [] as Set
            int totalPatches = 0
            Map<String, Integer> totalScaleStats = [:]
            if (continueMode) {
                allClasses.addAll((existingLog.get("classes") as List ?: []))
                totalPatches = (existingLog.get("totalPatches") as Number)?.intValue() ?: 0
                (existingLog.get("scaleStats") as Map ?: [:]).each { k, v ->
                    totalScaleStats[k as String] = (v as Number)?.intValue() ?: 0
                }
            }
            // Preserve existing slide entries; new runs replace or append per-slide
            List slideLog = continueMode
                ? ((existingLog.get("slides") as List ?: []).collect())
                : []

            double valFrac = valFractionSpinner.getValue()
            def entries    = project.getImageList()
            int total      = entries.size()

            for (int idx = 0; idx < entries.size(); idx++) {
                if (extractCancelled.get()) break

                def entry   = entries[idx]
                String sName = entry.getImageName()

                // Skip slides already in dataset unless marked redo
                if (continueMode && alreadyDone.contains(sName) && !redoSlides.contains(sName)) {
                    final int sn = idx + 1
                    Platform.runLater { appendExtractLog("[${sn}/${total}] ${sName} → already extracted (set status=redo to force)\n") }
                    continue
                }

                // Role: table takes priority, fall back to project metadata
                String rawRole = tableRoles.containsKey(sName)
                    ? tableRoles[sName]
                    : slideMetadataRole(entry)

                if (!rawRole || rawRole == "skip") {
                    final int sn = idx + 1
                    Platform.runLater { appendExtractLog("[${sn}/${total}] ${sName} → skip\n") }
                    continue
                }

                String role = rawRole
                if (role == "train" && autoValCheck.isSelected() && valFrac > 0 && Math.random() < valFrac)
                    role = "val"
                extractor.split = role

                final int slideNum = idx + 1; final String finalName = sName
                extractor.onPatchProgress = { int done, int exp ->
                    Platform.runLater {
                        extractProgress.setText("Slide ${slideNum}/${total}  [${finalName}]  patch ${done}/${exp}")
                    }
                }

                try {
                    def imageData = entry.readImageData()
                    def slideId   = sanitize(sName)
                    def allAnns   = imageData.getHierarchy().getAnnotationObjects()
                    def slideAnns = allAnns.findAll { it.getPathClass() != null && it.getPathClass().getName() != null }
                    slideAnns.each { allClasses << it.getPathClass().getName() }

                    extractor.datasetRoot = dataDir.absolutePath
                    def extractResult = extractor.extract(imageData as ImageData<BufferedImage>, slideId)
                    int n = (extractResult.total as Number)?.intValue() ?: 0
                    Map slideScaleStats = extractResult.scaleStats as Map ?: [:]
                    totalPatches += n
                    slideScaleStats.each { k, v ->
                        totalScaleStats[k as String] = (totalScaleStats[k as String] ?: 0) + ((v as Number)?.intValue() ?: 0)
                    }

                    // Replace prior log entry for this slide (if re-extracting), else append
                    slideLog.removeIf { (it as Map).get("slide") == sName }
                    slideLog << [slide: sName, role: role, patches: n,
                                 annotations: slideAnns.size(), scaleStats: slideScaleStats]

                    final int running     = totalPatches
                    final int patchCount  = n
                    final Map slideScales = slideScaleStats
                    final int unclassified = allAnns.size() - slideAnns.size()
                    Platform.runLater {
                        String msg
                        if (n > 0) {
                            def scaleSummary = slideScales.collect { k, v -> "${v}@${k}" }.join(", ")
                            msg = "[${slideNum}/${total}] ${sName} (${role}) → ${n} patches [${scaleSummary}]  (total: ${running})\n"
                        } else if (unclassified > 0 && slideAnns.isEmpty()) {
                            msg = "[${slideNum}/${total}] ${sName} (${role}) → 0 patches — ${unclassified} annotation(s) found but none have a class assigned. Assign a class in QuPath then re-extract.\n"
                        } else {
                            msg = "[${slideNum}/${total}] ${sName} (${role}) → 0 patches — no annotations found, status kept as 'new'\n"
                        }
                        appendExtractLog(msg)
                        extractProgress.setText("Slide ${slideNum}/${total} — patches so far: ${running}")
                        // Only mark done if patches were actually extracted
                        def matchRow = slideRoleEntries.find { it.name == sName }
                        if (matchRow) matchRow.status.set(patchCount > 0 ? "done" : "new")
                    }
                } catch (Exception e) {
                    Platform.runLater { appendExtractLog("  ERROR ${sName}: ${e.message}\n") }
                }
            }

            PatchExtractor.writeDatasetYaml(dataDir.absolutePath, allClasses.sort() as List)
            // Convert scale labels ("1×", "2×") back to doubles for inference use
            def usedDownsamples = totalScaleStats.keySet()
                .collect { it.replace("×", "").toDouble() }
                .findAll { !it.isNaN() }
                .sort()
            def logData = [date: todayStr(), slides: slideLog,
                classes: allClasses.sort(), totalPatches: totalPatches,
                scaleStats: totalScaleStats, downsamples: usedDownsamples]
            writeFile(new File(dataDir, "extract_log.json"), new Gson().toJson(logData))

            final String dataDirPath  = dataDir.absolutePath
            final int    fin          = totalPatches
            final boolean cancelled   = extractCancelled.get()
            final Map    finalScales  = totalScaleStats
            Platform.runLater {
                stopExtractBtn.setDisable(true)
                activeDatasetDir = dataDirPath
                activeDatasetLabel.setText("Active dataset: ${new File(dataDirPath).name}")
                activeDatasetLabel.setTooltip(new Tooltip(dataDirPath))
                openDatasetBtn.setDisable(false)
                extractProgress.setText(cancelled ? "Stopped — patches: ${fin}" : "Done — total: ${fin}")
                def scaleSummary = finalScales.collect { k, v -> "${v}@${k}" }.join(", ")
                appendExtractLog(cancelled
                    ? "\nStopped. Patches: ${fin}  Scales: ${scaleSummary}\n"
                    : "\nDone. Total patches: ${fin}  Scales: ${scaleSummary}\nClasses: ${allClasses.sort()}\ndataset.yaml written.\n")
                refreshSlideStatuses()
                slideTable.refresh()
                refreshDatasetList()
                refreshSplitChart(dataDirPath)
            }
        }
    }

    // ── Slide table helpers ───────────────────────────────────────────────────

    private void scanSlides() {
        def project = qupath.getProject()
        if (!project) { alert("Open a QuPath project first."); return }

        def latestDir = latestDataDir()
        Set<String> done = latestDir ? extractedSlideNames(latestDir) : [] as Set

        def rows = project.getImageList().collect { entry ->
            String name = entry.getImageName()
            String role = slideMetadataRole(entry) ?: "skip"
            def row     = new SlideRoleEntry(name, entry, role)
            row.status.set(done.contains(name) ? "done" : "new")
            return row
        }
        Platform.runLater {
            slideRoleEntries.setAll(rows)
            appendExtractLog("Scanned ${rows.size()} slides.\n")
            updateDatasetInfoLabel()
        }
    }

    private void populateRolesFromMetadata() {
        if (slideRoleEntries.isEmpty()) { scanSlides(); return }
        slideRoleEntries.each { row ->
            String role = slideMetadataRole(row.entry) ?: "skip"
            row.role.set(role)
        }
        Platform.runLater { slideTable.refresh() }
    }

    private void refreshSlideStatuses() {
        if (slideRoleEntries.isEmpty()) return
        def latestDir = latestDataDir()
        Set<String> done = latestDir ? extractedSlideNames(latestDir) : [] as Set
        slideRoleEntries.each { row -> row.status.set(done.contains(row.name) ? "done" : "new") }
    }

    private void populateTargetDatasetBox() {
        def base = new File(yoloAlRoot(), "data")
        if (!base.exists()) return
        def datasets = base.listFiles({ f -> f.isDirectory() && new File(f, "dataset.yaml").exists() } as FileFilter)
            ?.sort { it.lastModified() }
            ?.collect { it.absolutePath }
            ?: []
        Platform.runLater {
            targetDatasetBox.setItems(FXCollections.observableArrayList(datasets))
            // Pre-select the active dataset or the latest
            def preferred = activeDatasetDir ?: (datasets.isEmpty() ? null : datasets.last())
            if (preferred && datasets.contains(preferred)) targetDatasetBox.setValue(preferred)
            else if (!datasets.isEmpty()) targetDatasetBox.setValue(datasets.last())
            updateDatasetInfoLabel()
        }
    }

    private void updateDatasetInfoLabel() {
        if (appendRadio.isSelected()) {
            String selected = targetDatasetBox.getValue()
            if (selected) {
                int count = extractedSlideNames(new File(selected)).size()
                Platform.runLater {
                    datasetInfoLabel.setText("Adding to: ${new File(selected).name}  (${count} slides already extracted)")
                }
            } else {
                Platform.runLater { datasetInfoLabel.setText("No dataset selected — click ⟳ to scan") }
            }
        } else {
            Platform.runLater { datasetInfoLabel.setText("") }
        }
    }

    private void saveRoleToMetadata(SlideRoleEntry row) {
        try {
            row.entry.putMetadataValue("role", row.role.get())
            qupath.getProject()?.syncChanges()
        } catch (Exception e) {
            log.warning("Could not save role to metadata for ${row.name}: ${e.message}")
        }
    }

    private String slideMetadataRole(entry) {
        try { return (entry.getMetadataValue("role") ?: "").trim().toLowerCase() }
        catch (Exception ignored) {
            try { return (entry.getMetadata()?.get("role") ?: "").trim().toLowerCase() }
            catch (Exception ignored2) { return "" }
        }
    }

    private Set<String> extractedSlideNames(File dataDir) {
        if (!dataDir?.exists()) return [] as Set
        def logFile = new File(dataDir, "extract_log.json")
        if (!logFile.exists()) return [] as Set
        try {
            def log = gson.fromJson(logFile.text, mapType)
            return ((log.get("slides") as List ?: []).collect { (it as Map).get("slide") as String }).toSet()
        } catch (Exception e) { return [] as Set }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Actions — train
    // ═══════════════════════════════════════════════════════════════════════

    private void startTraining() {
        if (!client.isAlive()) { alert("Start the Python server first."); return }

        // Use selected dataset; fall back to active dir from last extraction, then latest on disk
        def dataDir = datasetSelectBox.getValue()?.trim()
            ?: activeDatasetDir
            ?: latestDataDir()?.absolutePath
        if (!dataDir) { alert("Select a dataset (click ⟳ in the Dataset selector)."); return }

        def yamlPath = new File(dataDir, "dataset.yaml")
        if (!yamlPath.exists()) { alert("dataset.yaml not found in:\n${dataDir}"); return }

        def modelDir = nextModelDir(modelBox.getValue())

        trainLossSeries.getData().clear(); valLossSeries.getData().clear()
        map50Series.getData().clear(); lastEpoch = 0
        trainBtn.setDisable(true); stopBtn.setDisable(false)
        trainStatus.setText("Starting…")

        runBg {
            client.setDataset(yamlPath.absolutePath)
            client.startTraining([
                model:     modelBox.getValue(),
                epochs:    epochsSpinner.getValue(),
                imgsz:     imgsSpinner.getValue(),
                batch:     batchSpinner.getValue(),
                lr0:       lrSpinner.getValue(),
                project:   modelDir.parent,
                name:      modelDir.name,
                degrees:   augDegrees.getValue(),
                translate: augTranslate.getValue(),
                scale:     augScale.getValue(),
                flipud:    augFlipud.getValue(),
                fliplr:    augFliplr.getValue(),
                mosaic:    augMosaic.getValue(),
                hsv_h:     augHsvH.getValue(),
                hsv_s:     augHsvS.getValue(),
                hsv_v:     augHsvV.getValue()
            ])
        }

        scheduler.scheduleAtFixedRate({
            try { def s = client.getStatus(); if (s) Platform.runLater { updateTrainUI(s, modelDir) } }
            catch (Exception ignored) {}
        }, 2, 2, TimeUnit.SECONDS)
    }

    private void stopTraining() { runBg { client.stopTraining() } }

    private void updateTrainUI(Map s, File modelDir) {
        def state  = s.get("state") as String ?: "unknown"
        int epoch  = (s.get("epoch")  as Number)?.intValue() ?: 0
        int epochs = (s.get("epochs") as Number)?.intValue() ?: epochsSpinner.getValue()

        trainStatus.setText("$state  epoch $epoch/$epochs")

        if (epoch > lastEpoch) {
            lastEpoch = epoch
            if (s.get("train_loss")) addPoint(trainLossSeries, epoch, (s.get("train_loss") as Number).doubleValue())
            if (s.get("val_loss"))   addPoint(valLossSeries,   epoch, (s.get("val_loss")   as Number).doubleValue())
            if (s.get("map50"))      addPoint(map50Series,      epoch, (s.get("map50")      as Number).doubleValue())
        }

        if (state in ["done", "stopped", "error"]) {
            trainBtn.setDisable(false); stopBtn.setDisable(true)
            scheduler.shutdownNow(); scheduler = Executors.newSingleThreadScheduledExecutor()

            if (state == "done") {
                runBg {
                    def m = client.getMetrics(modelDir.absolutePath)
                    // Save train log
                    def bestPt = new File(modelDir, "weights/best.pt")
                    activeModelPath = bestPt.exists() ? bestPt.absolutePath : null
                    writeFile(new File(modelDir, "train_log.json"), new Gson().toJson([
                        date: todayStr(), model: modelBox.getValue(),
                        dataset: activeDatasetDir, epochs: epochsSpinner.getValue(),
                        precision: m.get("precision"), recall: m.get("recall"),
                        map50: m.get("map50"), map50_95: m.get("map50_95")
                    ]))
                    Platform.runLater {
                        metricsLabel.setText(String.format(
                            "P=%.3f  R=%.3f  mAP50=%.3f  mAP50-95=%.3f",
                            (m.get("precision") as Number)?.doubleValue() ?: 0,
                            (m.get("recall")    as Number)?.doubleValue() ?: 0,
                            (m.get("map50")     as Number)?.doubleValue() ?: 0,
                            (m.get("map50_95")  as Number)?.doubleValue() ?: 0))
                        refreshModelList()
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Actions — infer
    // ═══════════════════════════════════════════════════════════════════════

    // ── Plan ──────────────────────────────────────────────────────────────

    private void planInference() {
        def project = qupath.getProject()
        if (!project) { alert("Open a QuPath project first."); return }

        String modelPath = modelSelectBox.getValue()?.trim() ?: activeModelPath
        int tileSize = tileSizeSpinner.getValue()
        int overlap  = tileOverlapSpinner.getValue()

        def testEntries = getTestEntries(project)
        List planSlides
        if (testEntries.isEmpty()) {
            def curData = qupath.getImageData()
            if (!curData) {
                alert("No slides with role=test, and no slide is currently open.\nSet metadata role=test on a slide, or open the slide you want to plan.")
                return
            }
            planSlides = [[name: curData.getServer()?.getMetadata()?.getName() ?: "current_slide",
                           imageData: curData]]
        } else {
            planSlides = testEntries.collect { entry -> [name: entry.getImageName(), imageData: null, entry: entry] }
        }

        def sb = new StringBuilder()
        sb << "═══════════════════════════════════════\n"
        sb << "  INFERENCE PLAN\n"
        sb << "═══════════════════════════════════════\n\n"
        sb << "Model  : ${modelPath ?: '(none selected)'}\n"
        sb << "Conf   : ${confSpinner.getValue()}   NMS IoU: ${iouSpinner.getValue()}\n"
        sb << "Tile   : ${tileSize}px   Overlap: ${overlap}px   Step: ${tileSize-overlap}px\n"
        sb << "Tissue : ≥ ${(tissueFracSpinner.getValue()*100).round()}% coverage per tile\n\n"

        planSlides.each { slide ->
            sb << "── ${slide.name} ─────────\n"
            try {
                def imageData = slide.imageData ?: slide.entry.readImageData()
                def server    = imageData.getServer()
                sb << "   Size    : ${server.getWidth()} × ${server.getHeight()} px\n"

                // Check for user ROIs — unclassified annotations only, matching actual inference logic
                def rois = imageData.getHierarchy().getAnnotationObjects()
                    .findAll { it.getROI() != null && it.getPathClass() == null }
                if (rois) {
                    sb << "   ROI mode: ${rois.size()} annotation(s) → polygon mask → tile within ROIs\n"
                    long tiles = rois.sum { ann ->
                        def r = ann.getROI()
                        long cols = (long)Math.ceil(r.getBoundsWidth()  / (tileSize - overlap))
                        long rows = (long)Math.ceil(r.getBoundsHeight() / (tileSize - overlap))
                        cols * rows
                    } as long
                    sb << "   Est. tiles: ~${tiles} (before tissue filter)\n"
                } else {
                    sb << "   ROI mode: none → tissue detection at 1/32 resolution\n"
                    long step = tileSize - overlap
                    long cols = (long)Math.ceil(server.getWidth()  / (double)step)
                    long rows = (long)Math.ceil(server.getHeight() / (double)step)
                    sb << "   Est. tiles: ~${cols * rows} (before tissue filter; ~${(int)(cols*rows*tissueFracSpinner.getValue())} expected to pass)\n"
                }
            } catch (Exception e) {
                sb << "   (Could not read slide: ${e.message})\n"
            }
            sb << "\n"
        }

        sb << "Output : ${yoloAlRoot().absolutePath}/inference/infer_XX_${todayStr()}/\n"
        sb << "         Per-slide .geojson + infer_log.json\n\n"
        sb << "Proceed? Click Run Inference to start."

        Platform.runLater {
            def ta = new TextArea(sb.toString())
            ta.setEditable(false)
            ta.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11;")
            ta.setPrefSize(560, 420)

            def dialog = new Dialog<ButtonType>()
            dialog.setTitle("Inference Plan")
            dialog.getDialogPane().setContent(ta)
            dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL)
            dialog.initOwner(qupath.getStage())
            dialog.showAndWait()
        }
    }

    // ── Run ───────────────────────────────────────────────────────────────

    private void runInference() {
        if (!client.isAlive()) { alert("Start the Python server first."); return }
        def project = qupath.getProject()
        if (!project) { alert("Open a QuPath project first."); return }

        String modelPath = modelSelectBox.getValue()?.trim() ?: activeModelPath
        if (!modelPath || !new File(modelPath).exists()) {
            alert("Select a trained model (click ⟳ to scan)."); return
        }

        def testEntries = getTestEntries(project)
        List slides
        if (testEntries.isEmpty()) {
            def curData = qupath.getImageData()
            if (!curData) {
                alert("No slides with role=test, and no slide is currently open.\nSet metadata role=test on a slide, or open the slide you want to infer.")
                return
            }
            def curPath  = curData.getServer()?.getPath()
            def curEntry = project.getImageList().find {
                try { it.getServerPath() == curPath } catch (Exception ignored) { false }
            }
            slides = [[name: curData.getServer()?.getMetadata()?.getName() ?: "current_slide",
                       imageData: curData, entry: curEntry]]
        } else {
            slides = testEntries.collect { entry ->
                [name: entry.getImageName(), imageData: null, entry: entry]
            }
        }

        inferLog.clear(); inferBtn.setDisable(true); planBtn.setDisable(true)

        double conf        = confSpinner.getValue()
        double iouNms      = iouSpinner.getValue()
        int    tileSize    = tileSizeSpinner.getValue()
        int    overlap     = tileOverlapSpinner.getValue()
        double tissueFrac  = tissueFracSpinner.getValue()
        def    inferDir    = nextInferDir()

        appendInferLog("═══ Inference Run ═══════════════════════\n")
        appendInferLog("Model  : ${modelPath}\n")
        appendInferLog("Conf   : ${conf}   NMS IoU: ${iouNms}\n")
        appendInferLog("Tile   : ${tileSize}px  Overlap: ${overlap}px\n")
        appendInferLog("Output : ${inferDir.absolutePath}\n")
        appendInferLog("═════════════════════════════════════════\n\n")

        runBg {
            def inferLogData = [date: todayStr(), model: modelPath,
                conf: conf, iou: iouNms, tileSize: tileSize, overlap: overlap, slides: []]

            slides.each { slide ->
                def slideName = slide.name as String
                def slideId   = sanitize(slideName)
                Platform.runLater { appendInferLog("\n── ${slideName} ──\n") }

                try {
                    // Prefer live in-memory data so unsaved ROI annotations are included.
                    // Use case-insensitive, slash-normalised path comparison (Bio-Formats URIs
                    // can differ in case and separator on Windows).
                    def imageData = slide.imageData
                    if (!imageData) {
                        def liveData = qupath.getImageData()
                        String entryPath = ""; try { entryPath = slide.entry?.getServerPath() ?: "" } catch (Exception ignored) {}
                        String livePath  = ""; try { livePath  = liveData?.getServer()?.getPath() ?: "" } catch (Exception ignored) {}
                        boolean sameSlide = livePath && entryPath &&
                            livePath.replace('\\', '/').equalsIgnoreCase(entryPath.replace('\\', '/'))
                        imageData = sameSlide ? liveData : slide.entry?.readImageData()
                        if (!sameSlide && liveData) {
                            log.warning("Live data path did not match entry path — using saved data.\n  live:  ${livePath}\n  entry: ${entryPath}")
                        }
                    }
                    def server    = imageData.getServer()
                    def hierarchy = imageData.getHierarchy()

                    // Resolve local file path from server URI
                    String slidePath = null
                    try {
                        def uris = server.getURIs()
                        if (uris && !uris.isEmpty())
                            slidePath = new java.io.File(new java.net.URI(uris.first().toString())).absolutePath
                    } catch (Exception e) { log.warning("URI→path: $e.message") }

                    if (!slidePath || !new File(slidePath).exists()) {
                        Platform.runLater { appendInferLog("  ERROR: Cannot resolve local file path for this slide.\n") }
                        return
                    }

                    // User-drawn ROI polygons — unclassified annotations only
                    def unclassifiedAnns = hierarchy.getAnnotationObjects()
                        .findAll { it.getROI() != null && it.getPathClass() == null }
                    def roiPolygons = unclassifiedAnns.collect { ann ->
                        ann.getROI().getAllPoints().collect { pt -> [pt.getX(), pt.getY()] }
                    }.findAll { it.size() >= 3 }
                    final int roiCount = roiPolygons.size()
                    final int totalAnns = hierarchy.getAnnotationObjects().size()
                    Platform.runLater {
                        appendInferLog("  Annotations: ${totalAnns} total, ${unclassifiedAnns.size()} unclassified ROI(s) → ${roiCount} valid polygon(s)\n")
                    }

                    Platform.runLater {
                        appendInferLog("  Mode : ${roiPolygons ? "${roiPolygons.size()} user ROI(s)" : "auto tissue detection"}\n")
                        appendInferLog("  Starting…\n")
                    }

                    // Progress poller — runs alongside the blocking inferWSI call
                    def inferDone = new java.util.concurrent.atomic.AtomicBoolean(false)
                    Thread.start {
                        while (!inferDone.get()) {
                            Thread.sleep(1500)
                            if (inferDone.get()) break
                            try {
                                def prog = client.getInferProgress()
                                if (prog?.get("active")) {
                                    int done  = (prog.get("done")  as Number)?.intValue() ?: 0
                                    int total = (prog.get("total") as Number)?.intValue() ?: 0
                                    int dets  = (prog.get("dets")  as Number)?.intValue() ?: 0
                                    int pct   = total > 0 ? (int)(done * 100.0 / total) : 0
                                    Platform.runLater {
                                        extractProgress.setText("Inferring: ${done}/${total} tiles (${pct}%)  ${dets} raw dets")
                                    }
                                }
                            } catch (Exception ignored) {}
                        }
                    }

                    // Delegate entire WSI pipeline to Python
                    def result = client.inferWSI([
                        slide_path:   slidePath,
                        model_path:   modelPath,
                        conf:         conf,
                        iou:          iouNms,
                        tile_size:    tileSize,
                        overlap:      overlap,
                        tissue_frac:  tissueFrac,
                        output_dir:   inferDir.absolutePath,
                        roi_polygons: roiPolygons,
                        debug_tiles:  debugTilesChk.isSelected(),
                        downsamples:  selectedDownsamples()
                    ])
                    inferDone.set(true)

                    if (result?.get("error")) {
                        Platform.runLater { appendInferLog("  ERROR: ${result.get('error')}\n") }
                        return
                    }

                    int tilesTotal = (result?.get("tiles_total") as Number)?.intValue() ?: 0
                    int tilesUsed  = (result?.get("tiles_used")  as Number)?.intValue() ?: 0
                    int detCount   = (result?.get("detections")  as Number)?.intValue() ?: 0
                    String gjPath  = result?.get("geojson_path") as String

                    Map confHist    = result?.get("conf_hist")    as Map ?: [:]
                    Map classCounts = result?.get("class_counts") as Map ?: [:]
                    Platform.runLater {
                        appendInferLog("  Tiles : ${tilesUsed} tissue / ${tilesTotal} total\n")
                        appendInferLog("  Detections: ${detCount} (after cross-tile NMS)\n")
                        if (confHist) {
                            def histStr = confHist.collect { k, v -> "${k}: ${(v as Number).intValue()}" }.join("  ")
                            appendInferLog("  Conf  : ${histStr}\n")
                        }
                        if (classCounts) {
                            def clsStr = classCounts.collect { k, v -> "${k}=${(v as Number).intValue()}" }.join("  ")
                            appendInferLog("  Class : ${clsStr}\n")
                        }
                    }

                    // Import GeoJSON into QuPath hierarchy
                    if (gjPath && new File(gjPath).exists()) {
                        List newObjects = []
                        String importErr = null
                        try {
                            def fc = gson.fromJson(new FileReader(new File(gjPath)), Map.class)
                            def plane = defaultPlane()
                            ((fc?.get("features") as List) ?: []).each { f ->
                                try {
                                    def coords = (((f as Map).get("geometry") as Map).get("coordinates") as List)[0] as List
                                    def xs = coords.collect { (it as List)[0] as double }
                                    def ys = coords.collect { (it as List)[1] as double }
                                    double rx = xs.min(), ry = ys.min()
                                    double rw = xs.max() - rx, rh = ys.max() - ry
                                    def props   = (f as Map).get("properties") as Map
                                    def clsName = (props?.get("classification") as Map)?.get("name") as String
                                    def pc  = clsName ? PathClass.fromString(clsName) : null
                                    def roi = ROIs.createRectangleROI(rx, ry, rw, rh, plane)
                                    newObjects << PathObjects.createDetectionObject(roi, pc)
                                } catch (Exception fe) { log.warning("GeoJSON feature skip: $fe.message") }
                            }
                        } catch (Exception e) { importErr = e.message }

                        if (newObjects) {
                            hierarchy.addObjects(newObjects)
                            try { slide.entry?.saveImageData(imageData) } catch (Exception e) { log.warning("Save: $e.message") }
                        }
                        final int n = newObjects.size(); final String err = importErr; final String gj = gjPath
                        Platform.runLater {
                            if (n > 0) hierarchy.fireHierarchyChangedEvent(null)
                            appendInferLog("  ✓ ${n} detections loaded into QuPath\n")
                            if (err) appendInferLog("  Import error: ${err}\n")
                            else if (n == 0) appendInferLog("  No detections — GeoJSON: ${gj}\n")
                        }
                    } else if (result?.get("warning")) {
                        Platform.runLater { appendInferLog("  ⚠ ${result.get('warning')}\n") }
                    }

                    inferLogData.slides << [slide: slideName, tiles: tilesUsed,
                        detections: detCount, geojson: gjPath]

                } catch (Exception e) {
                    Platform.runLater { appendInferLog("  ERROR: ${e.message}\n") }
                    log.log(java.util.logging.Level.WARNING, "Inference error on ${slideName}", e)
                }
            }

            writeFile(new File(inferDir, "infer_log.json"), new Gson().toJson(inferLogData))

            Platform.runLater {
                appendInferLog("\n═══ Done ════════════════════════════════\n")
                appendInferLog("Results  : ${inferDir.absolutePath}\n")
                appendInferLog("GeoJSON  : drag-drop onto QuPath viewer\n")
                inferBtn.setDisable(false); planBtn.setDisable(false)
            }
        }
    }

    // ── WSI Inference Helpers ─────────────────────────────────────────────

    private List getTestEntries(project) {
        project.getImageList().findAll { entry ->
            String role = ""
            try { role = (entry.getMetadataValue("role") ?: "").trim().toLowerCase() }
            catch (Exception ignored) {
                try { role = (entry.getMetadata()?.get("role") ?: "").trim().toLowerCase() }
                catch (Exception ignored2) {}
            }
            return role == "test"
        }
    }

    /**
     * Returns tiling regions for a slide.
     * Priority:
     *   1. Unlabeled annotations (no PathClass) drawn by the user → use as explicit ROI
     *   2. Fallback → Python tissue detection on a low-res thumbnail
     *
     * Training annotations (which always have a PathClass) are intentionally ignored.
     * To define an inference ROI: draw a rectangle/polygon in QuPath WITHOUT assigning a class.
     */
    private List<java.awt.Rectangle> getInferenceRegions(server, hierarchy) {
        def roiAnnotations = hierarchy.getAnnotationObjects()
            .findAll { it.getROI() != null && it.getPathClass() == null }

        if (roiAnnotations) {
            Platform.runLater { appendInferLog("  ROI mode : ${roiAnnotations.size()} user-drawn ROI(s)\n") }
            return roiAnnotations.collect { ann ->
                def r = ann.getROI()
                new java.awt.Rectangle(
                    (int) r.getBoundsX(), (int) r.getBoundsY(),
                    (int) r.getBoundsWidth(), (int) r.getBoundsHeight())
            }
        }

        // No explicit ROI → tissue detection via Python
        Platform.runLater { appendInferLog("  ROI mode : tissue detection (no unlabeled annotations found)\n") }
        return detectTissueBoundsViaPython(server)
    }

    /**
     * Detects tissue bounding box(es) by sending a low-res thumbnail to the Python server.
     * Returns a list of rectangles in full-resolution slide coordinates.
     * Falls back to the full slide if the server call fails.
     */
    private List<java.awt.Rectangle> detectTissueBoundsViaPython(server) {
        double thumbFactor = 32.0
        int thumbW = Math.max(1, (int)(server.getWidth()  / thumbFactor))
        int thumbH = Math.max(1, (int)(server.getHeight() / thumbFactor))

        // Read thumbnail
        def req = RegionRequest.createInstance(
            server.getPath(), thumbFactor, 0, 0, server.getWidth(), server.getHeight())
        BufferedImage thumb
        try { thumb = server.readRegion(req) }
        catch (Exception e) {
            log.warning("Thumbnail read failed: $e.message")
            return [new java.awt.Rectangle(0, 0, server.getWidth(), server.getHeight())]
        }

        // Encode as base64 PNG
        String b64
        try {
            def baos = new java.io.ByteArrayOutputStream()
            javax.imageio.ImageIO.write(thumb, "PNG", baos)
            b64 = java.util.Base64.getEncoder().encodeToString(baos.toByteArray())
        } catch (Exception e) {
            log.warning("Thumbnail encode failed: $e.message")
            return [new java.awt.Rectangle(0, 0, server.getWidth(), server.getHeight())]
        }

        // Ask Python server for tissue bounds
        try {
            def bounds = client.getTissueBounds(b64, thumbW, thumbH,
                server.getWidth(), server.getHeight())
            return bounds.collect { b ->
                def coords = b as List
                new java.awt.Rectangle(
                    (coords[0] as int), (coords[1] as int),
                    ((coords[2] as int) - (coords[0] as int)),
                    ((coords[3] as int) - (coords[1] as int)))
            }
        } catch (Exception e) {
            log.warning("Tissue detection failed: $e.message")
            return [new java.awt.Rectangle(0, 0, server.getWidth(), server.getHeight())]
        }
    }

    /**
     * Returns true if the tile has at least `threshold` fraction of non-white pixels.
     */
    private static boolean hasTissue(BufferedImage tile, double threshold) {
        int total   = tile.getWidth() * tile.getHeight()
        int tissue  = 0
        int needed  = (int)(total * threshold)
        for (int y = 0; y < tile.getHeight(); y++) {
            for (int x = 0; x < tile.getWidth(); x++) {
                int rgb = tile.getRGB(x, y)
                int r = (rgb >> 16) & 0xFF
                int g = (rgb >> 8)  & 0xFF
                int b =  rgb        & 0xFF
                if (r < 220 || g < 220 || b < 220) { tissue++; if (tissue >= needed) return true }
            }
        }
        return tissue >= needed
    }

    /**
     * Non-maximum suppression across tile boundaries.
     * Sorted by confidence descending; suppresses lower-conf overlapping boxes of same class.
     */
    private static List<Map> applyNMS(List<Map> dets, double iouThresh) {
        if (dets.isEmpty()) return []
        def sorted    = dets.sort(false) { -(it.conf as double) }
        def suppress  = new boolean[sorted.size()]
        def result    = []
        sorted.eachWithIndex { d, i ->
            if (suppress[i]) return
            result << d
            sorted.eachWithIndex { other, j ->
                if (j <= i || suppress[j] || d.cls != other.cls) return
                if (iouCalc(d, other) > iouThresh) suppress[j] = true
            }
        }
        return result
    }

    private static double iouCalc(Map a, Map b) {
        double ix1 = Math.max(a.x1 as double, b.x1 as double)
        double iy1 = Math.max(a.y1 as double, b.y1 as double)
        double ix2 = Math.min(a.x2 as double, b.x2 as double)
        double iy2 = Math.min(a.y2 as double, b.y2 as double)
        double inter = Math.max(0, ix2-ix1) * Math.max(0, iy2-iy1)
        if (inter == 0) return 0
        double aA = (a.x2 as double - a.x1 as double) * (a.y2 as double - a.y1 as double)
        double bA = (b.x2 as double - b.x1 as double) * (b.y2 as double - b.y1 as double)
        double union = aA + bA - inter
        return union > 0 ? inter / union : 0
    }

    private static void writeGeoJSON(File file, List<Map> dets, double confThreshold) {
        def features = dets.collect { d ->
            boolean uncertain = (d.conf as double) < confThreshold
            int colorRGB = uncertain ? java.awt.Color.RED.getRGB() : java.awt.Color.GREEN.getRGB()
            [type: "Feature",
             geometry: [type: "Polygon",
                coordinates: [[[d.x1,d.y1],[d.x2,d.y1],[d.x2,d.y2],[d.x1,d.y2],[d.x1,d.y1]]]],
             properties: [objectType: "detection",
                 classification: [name: d.cls as String, colorRGB: colorRGB],
                 name: String.format("%.2f", d.conf as double)]]
        }
        writeFile(file, new Gson().toJson([type: "FeatureCollection", features: features]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Utilities
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Gets the default ImagePlane (z=0, t=0) without an explicit import.
     * Tries several known package locations across QuPath versions,
     * then falls back to extracting it from a RegionRequest.
     */
    private static Object defaultPlane() {
        // Try known package locations
        for (String cls : [
                "qupath.lib.images.servers.ImagePlane",
                "qupath.lib.roi.ImagePlane",
                "qupath.lib.images.ImagePlane"]) {
            try { return Class.forName(cls).getMethod("getDefaultPlane").invoke(null) }
            catch (ClassNotFoundException ignored) {}
        }
        // Fallback: extract from a RegionRequest (always available)
        try {
            def req = RegionRequest.createInstance("fallback", 1.0, 0, 0, 1, 1)
            return req.getClass().getMethod("getImagePlane").invoke(req)
        } catch (Exception ignored) {}
        return null
    }

    private void addPoint(XYChart.Series s, int x, double y) {
        s.getData().add(new XYChart.Data<>(x, y))
    }

    private void appendExtractLog(String msg) { extractLog.appendText(msg) }
    private void appendInferLog(String msg)   { inferLog.appendText(msg) }
    private void setStatus(String msg)        { Platform.runLater { statusLabel.setText(msg) } }

    private void alert(String msg) {
        Platform.runLater { new Alert(Alert.AlertType.WARNING, msg, ButtonType.OK).showAndWait() }
    }

    private void runBg(Closure c) { Thread.start { c.call() } }

    private String sanitize(String name) { name.replaceAll(/[^A-Za-z0-9_\-]/, "_") }

    private static void writeFile(File f, String content) {
        Files.write(f.toPath(), content.getBytes("UTF-8"))
    }

    private String locatePython() {
        def base = locateExtensionDir()
        for (c in ["yolo-venv/Scripts/python.exe", "yolo-venv/bin/python", "yolo-venv/bin/python3"]) {
            def f = new File(base, c); if (f.exists()) return f.absolutePath
        }
        return "python3"
    }

    private String locateServerScript() {
        def base = locateExtensionDir()
        def f = new File(base, "yolo_server.py")
        return f.exists() ? f.absolutePath : new File(base, "python/yolo_server.py").absolutePath
    }

    private File locateExtensionDir() {
        File base = null
        try { base = qupath.getUserDataDirectory() } catch (Exception ignored) {}
        if (!base) {
            String appData = System.getenv("APPDATA")
            base = appData ? new File(appData, "QuPath") : new File(System.getProperty("user.home"), ".qupath")
        }
        def d = new File(base, "yolo-al"); d.mkdirs(); return d
    }

    private void showAbout() {
        Platform.runLater {
            def content = new javafx.scene.layout.VBox(10)
            content.setPadding(new Insets(12))
            content.setAlignment(Pos.CENTER_LEFT)

            def title = new Label("QuPath YOLO Active Learning")
            title.setStyle("-fx-font-size: 14; -fx-font-weight: bold;")

            def sub = new Label("Active learning for whole-slide image analysis\nusing YOLO object detection")
            sub.setStyle("-fx-font-size: 11; -fx-text-fill: #555;")
            sub.setWrapText(true)

            def sep = new Separator()

            def author = new Label("Author:   Sandeep Manandhar, PhD")
            author.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11;")
            def email = new Label("Contact:  manandhar.sandeep@gmail.com")
            email.setStyle("-fx-font-family: Monospaced; -fx-font-size: 11;")

            def sep2 = new Separator()

            def stack = new Label(
                "QuPath 0.6 · YOLOv8 / YOLO11 · FastAPI · Ultralytics\n" +
                "Patch extraction · Training · WSI inference · Active learning loop")
            stack.setStyle("-fx-font-size: 10; -fx-text-fill: #777;")
            stack.setWrapText(true)

            content.getChildren().addAll(title, sub, sep, author, email, sep2, stack)

            def dialog = new Dialog<ButtonType>()
            dialog.setTitle("About")
            dialog.getDialogPane().setContent(content)
            dialog.getDialogPane().getButtonTypes().add(ButtonType.OK)
            dialog.getDialogPane().setPrefWidth(380)
            dialog.initOwner(qupath.getStage())
            dialog.showAndWait()
        }
    }

    void shutdown() { scheduler.shutdownNow(); serverProcess?.destroy() }
}
