function submit_facefusion_task() {
    const res = Array.from(arguments);
    res[0] = randomId();
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
