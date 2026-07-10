import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "QuantFunc.TransformerFilter",

    nodeCreated(node) {
        if (node.comfyClass !== "QuantFuncModelAutoLoader") return;

        // MUST byte-match Python model_auto_loader.AUTO_DETECT ("[auto-detect]").
        // It is the node's transformer DEFAULT and always the FIRST option from
        // get_transformer_options(); at run time it resolves to the best weight the
        // GPU (CUDA device 0) can run.
        const AUTO_DETECT = "[auto-detect]";

        const seriesWidget = node.widgets.find(w => w.name === "model_series");
        const transformerWidget = node.widgets.find(w => w.name === "transformer");
        if (!seriesWidget || !transformerWidget) return;

        // Full option set as reported by the server (/object_info), captured
        // BEFORE any narrowing so every filter pass starts from the complete
        // list (never from a previously-narrowed one). allOptions[0] is
        // "[auto-detect]" (Python get_transformer_options prepends it), so the
        // order-preserving filter below keeps it FIRST.
        const allOptions = [...transformerWidget.options.values];

        // preserveCurrent=true  -> keep the currently-selected value valid even
        //                          if it doesn't match the series (workflow load
        //                          restores the value; dropping it would make
        //                          ComfyUI flag an already-present model as
        //                          "missing" and wipe the user's choice).
        // preserveCurrent=false -> manual series change: reset a now-mismatched
        //                          selection back to "[auto-detect]" (the node's
        //                          default → the engine auto-picks the best weight
        //                          for the GPU), NOT "None".
        function filterTransformer(preserveCurrent) {
            const series = seriesWidget.value || "";
            // "QuantFunc/Klein-9B-Series" -> "Klein-9B-Series"
            const shortName = series.includes("/") ? series.split("/").pop() : series;

            // Keep the two sentinels ([auto-detect], None) + this series' weights.
            // filter() preserves allOptions' order, so [auto-detect] stays FIRST.
            const filtered = allOptions.filter(opt =>
                opt === AUTO_DETECT || opt === "None" || opt.startsWith(shortName + "/")
            );
            // Defensive only (normally unreachable — [auto-detect] is always in
            // allOptions and always kept above): guarantee the default is present
            // and first even if the server list were ever malformed.
            if (!filtered.includes(AUTO_DETECT)) filtered.unshift(AUTO_DETECT);

            const current = transformerWidget.value;
            // The sentinels ([auto-detect]/None) are always in `filtered`, so only a
            // real, now-mismatched weight can reach the branch below.
            if (current && current !== "None" && current !== AUTO_DETECT
                    && !filtered.includes(current)) {
                if (preserveCurrent) {
                    filtered.push(current);              // keep restored value valid
                } else {
                    transformerWidget.value = AUTO_DETECT; // reset after a manual series switch
                }
            }

            transformerWidget.options.values = filtered;
        }

        // Manual series change: re-filter and reset a mismatched selection.
        const origCallback = seriesWidget.callback;
        seriesWidget.callback = function (...args) {
            const r = origCallback ? origCallback.apply(this, args) : undefined;
            filterTransformer(false);
            return r;
        };

        // nodeCreated runs BEFORE configure() restores saved widget values, so
        // re-filter once the workflow has restored model_series + transformer.
        // Without this, options stay narrowed to the DEFAULT series and the
        // restored value (e.g. a Klein-9B transformer) is reported as a
        // "missing model" on every load.
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const r = origOnConfigure ? origOnConfigure.apply(this, args) : undefined;
            filterTransformer(true);
            return r;
        };

        // Initial pass (fresh node: value is the default "[auto-detect]").
        filterTransformer(true);
    }
});
