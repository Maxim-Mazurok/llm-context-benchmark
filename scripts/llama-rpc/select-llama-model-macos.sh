#!/usr/bin/env bash

recommended_hugging_face_repository='bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q5_K_L'

resolve_llama_model_selection() {
    if [[ -n "$model_path" || -n "$hugging_face_repository" ]]; then
        return
    fi

    if [[ ! -t 0 ]]; then
        hugging_face_repository="$recommended_hugging_face_repository"
        return
    fi

    local discovered_model_path
    local model_directory
    local selected_model
    local -a discovered_model_paths=()
    local -a model_directories=(
        "$HOME/.omlx/models"
        "$HOME/.lmstudio/models"
    )

    while IFS= read -r discovered_model_path; do
        discovered_model_paths+=("$discovered_model_path")
    done < <(
        for model_directory in "${model_directories[@]}"; do
            if [[ -d "$model_directory" ]]; then
                find "$model_directory" -type f -name '*.gguf'
            fi
        done | sort
    )

    if (( ${#discovered_model_paths[@]} == 0 )); then
        printf 'No downloaded GGUF models found under OMLX or LM Studio; using %s.\n' \
            "$recommended_hugging_face_repository" >&2
        hugging_face_repository="$recommended_hugging_face_repository"
        return
    fi

    printf 'Select a llama.cpp model:\n' >&2
    PS3='Model number: '
    select selected_model in \
        "${discovered_model_paths[@]}" \
        "Download recommended: $recommended_hugging_face_repository"; do
        if [[ -z "$selected_model" ]]; then
            printf 'Invalid selection.\n' >&2
            continue
        fi

        if [[ "$selected_model" == "Download recommended: $recommended_hugging_face_repository" ]]; then
            hugging_face_repository="$recommended_hugging_face_repository"
        else
            model_path="$selected_model"
        fi
        return
    done
}