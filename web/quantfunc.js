import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "QuantFunc.TransformerFilter",

    nodeCreated(node) {
        if (node.comfyClass !== "QuantFuncModelAutoLoader") return;

        const seriesWidget = node.widgets.find(w => w.name === "model_series");
        const transformerWidget = node.widgets.find(w => w.name === "transformer");
        if (!seriesWidget || !transformerWidget) return;

        // Save the full list of options
        const allOptions = [...transformerWidget.options.values];

        function filterTransformer() {
            const series = seriesWidget.value || "";
            // Extract short name: "QuantFunc/Qwen-Image-Edit-Series" -> "Qwen-Image-Edit-Series"
            const shortName = series.includes("/") ? series.split("/").pop() : series;

            const filtered = allOptions.filter(opt =>
                opt === "None" || opt.startsWith(shortName + "/")
            );

            transformerWidget.options.values = filtered.length > 0 ? filtered : ["None"];

            // Reset selection if current value is not in filtered list
            if (!filtered.includes(transformerWidget.value)) {
                transformerWidget.value = "None";
            }
        }

        // Filter on series change
        const origCallback = seriesWidget.callback;
        seriesWidget.callback = function(...args) {
            if (origCallback) origCallback.apply(this, args);
            filterTransformer();
        };

        // Initial filter
        filterTransformer();
    }
});
