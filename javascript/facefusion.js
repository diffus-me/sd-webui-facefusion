async function submit_facefusion_task() {
    addGenerateGtagEvent("#facefusion_start_button > span", "facefusion_generation_button");
    await tierCheckButtonInternal("FaceFusion");

    const res = Array.from(arguments);
    const source_image = res[1];
    const target_image = res[2];
    const target_video = res[3];
    if (source_image && (target_image || target_video)) {
        res[0] = randomId();
    } else {
        res[0] = "";
    }
    return res;
}

onUiLoaded(async function () {
    const tab_name = "facefusion_interface";
    systemMonitorState[tab_name] = {
        generate_button_id: "facefusion_start_button",
        timeout_id: null,
        functions: {
            "extensions.facefusion": {
                params: {
                    width: 512,
                    height: 512,
                    n_iter: 1,
                },
                link_params: {}, // tab_name: function_name
                mutipliers: {}, // multipler_name: value
                link_mutipliers: {}, // function_name: param_name
            },
        },
    };
    await updateButton(tab_name);
});
