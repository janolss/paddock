//! Native download commands reference immutable catalog IDs, never paths/URLs.
//! Transfers, integrity, admission and recovery are the web manager's registry.
use paddock_manager::registry::{ArtifactKind, Registry};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Command {
    List {},
    Plan {
        model: String,
        artifact: String,
    },
    Pull {
        model: String,
        artifact: String,
        selection: Vec<String>,
    },
    Pause {
        id: String,
    },
    Resume {
        id: String,
    },
    Delete {
        model: String,
        artifact: String,
    },
}

fn selection(registry: &Registry, model: &str, artifact: &str) -> Result<Vec<String>, String> {
    if model.len() > 200 || artifact.len() > 100 {
        return Err("Invalid model selection".into());
    }
    let entry = registry
        .catalog_of(model)
        .ok_or("Select a model from the library")?;
    let weights = entry
        .artifact(artifact)
        .ok_or("Select a weights option from the library")?;
    // Split encoder towers can be fetched independently from native model
    // settings. Only catalog-reviewed companions of a Metal weights artifact
    // qualify; this never accepts a URL, path or a CUDA-only companion.
    if entry.split_towers()
        && matches!(weights.kind, ArtifactKind::Vision | ArtifactKind::Audio)
        && weights.runtime.supports_backend("metal")
        && entry.artifacts.iter().any(|a| {
            a.kind == ArtifactKind::Weights
                && a.runtime.supports_backend("metal")
                && a.runtime.allows_companion(&weights.id)
        })
    {
        return Ok(vec![weights.id.clone()]);
    }
    if weights.kind != ArtifactKind::Weights || !weights.runtime.supports_backend("metal") {
        return Err("These weights cannot run on Metal".into());
    }
    let mut ids = vec![weights.id.clone()];
    ids.extend(
        entry
            .artifacts
            .iter()
            .filter(|a| {
                a.kind != ArtifactKind::Weights
                    && (a.default || a.required)
                    && weights.runtime.allows_companion(&a.id)
                    && a.runtime.supports_backend("metal")
            })
            .map(|a| a.id.clone()),
    );
    Ok(ids)
}

pub(crate) fn execute(registry: &Registry, command: Command) -> Result<String, String> {
    let mut result = json!({});
    match command {
        Command::List {} => {}
        Command::Plan { model, artifact } => {
            let ids = selection(registry, &model, &artifact)?;
            let entry = registry.catalog_of(&model).ok_or("Model disappeared")?;
            let mut seen = std::collections::HashSet::new();
            let files: Vec<_> = ids
                .iter()
                .filter_map(|id| entry.artifact(id))
                .flat_map(|a| &a.files)
                .filter(|f| seen.insert(f.dest.clone()))
                .collect();
            let total: u64 = files.iter().map(|f| f.size).sum();
            let remaining: u64 = files
                .iter()
                .filter(|f| {
                    !std::fs::metadata(registry.models_dir().join(&f.dest))
                        .is_ok_and(|m| m.len() == f.size)
                })
                .map(|f| f.size)
                .sum();
            let volume = registry
                .models_dir()
                .ancestors()
                .find(|p| p.exists())
                .ok_or("Model storage volume unavailable")?;
            let free = paddock_manager::registry::disk_free(volume);
            let disk_need = registry
                .download_disk_need(&model, &ids)
                .map_err(|e| e.to_string())?;
            result["plan"] = json!({"model": model, "artifact": artifact, "display": entry.display,
                "selection": ids, "total": total, "remaining": remaining, "free": free, "disk_need": disk_need,
                "file_count": files.len(), "pieces": ids.iter().filter_map(|id| entry.artifact(id))
                    .map(|a| json!({"id": a.id, "label": a.label, "size": a.total_size()})).collect::<Vec<_>>() });
        }
        Command::Pull {
            model,
            artifact,
            selection: expected,
        } => {
            let ids = selection(registry, &model, &artifact)?;
            if ids != expected {
                return Err("The download selection changed. Review it again.".into());
            }
            result["job"] = registry
                .start_pull(&model, Some(&ids))
                .map_err(|e| e.to_string())?
                .into();
        }
        Command::Pause { id } => {
            validate_job(registry, &id)?;
            if !registry.cancel_pull(&id) {
                return Err("This download is no longer active. Refresh Downloads.".into());
            }
        }
        Command::Resume { id } => {
            validate_job(registry, &id)?;
            result["job"] = registry.resume_pull(&id).map_err(|e| e.to_string())?.into();
        }
        Command::Delete { model, artifact } => {
            let ids = selection(registry, &model, &artifact)?;
            let deleted = registry
                .delete_selection(&model, &ids)
                .map_err(|e| e.to_string())?;
            result["deleted"] = json!({
                "model": deleted.model,
                "artifact": deleted.artifact,
                "removed_files": deleted.removed_files,
                "freed_bytes": deleted.freed_bytes,
                "kept_shared_files": deleted.kept_shared_files,
            });
        }
    }
    result["jobs"] = Value::Array(
        registry
            .jobs()
            .iter()
            .map(|job| {
                let mut row = job.snapshot();
                // Follow plans belong to the web host. Native never projects or replays
                // raw spawn specs or credentials, including on restored history.
                row.as_object_mut()
                    .expect("a download snapshot is a JSON object")
                    .remove("start");
                row["weights"] = registry
                    .catalog_of(&job.model_id)
                    .and_then(|model| {
                        job.artifacts.as_ref()?.iter().find_map(|id| {
                            let a = model.artifact(id)?;
                            (a.kind == ArtifactKind::Weights && a.runtime.supports_backend("metal"))
                                .then(|| a.id.clone())
                        })
                    })
                    .into();
                row
            })
            .collect(),
    );
    let json = result.to_string();
    if json.len() > 1024 * 1024 {
        return Err("Download history exceeds its native projection limit".into());
    }
    Ok(json)
}

fn validate_job(registry: &Registry, id: &str) -> Result<(), String> {
    if id.len() != 32 || !id.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Invalid download identity".into());
    }
    let job = registry.job(id).ok_or("Download not found")?;
    let model = registry
        .catalog_of(&job.model_id)
        .ok_or("This model is no longer in the catalog")?;
    if job.artifacts.as_ref().is_none_or(|ids| {
        ids.iter().any(|id| {
            model
                .artifact(id)
                .is_none_or(|a| !a.runtime.supports_backend("metal"))
        })
    }) {
        return Err("This download belongs to another backend".into());
    }
    if job
        .follow
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
    {
        return Err("Manage this queued web-server start in web Studio".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embeddinggemma2_native_downloads_separate_optional_audio_from_picture_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::new(dir.path().into()).with_backend("metal");
        assert_eq!(
            selection(&registry, "embeddinggemma-2", "q8").unwrap(),
            ["q8", "vision"]
        );
        for tower in ["vision", "audio"] {
            assert_eq!(
                selection(&registry, "embeddinggemma-2", tower).unwrap(),
                [tower]
            );
            let reply: Value = serde_json::from_str(
                &execute(
                    &registry,
                    Command::Plan {
                        model: "embeddinggemma-2".into(),
                        artifact: tower.into(),
                    },
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(reply["plan"]["file_count"], 1);
            assert_eq!(reply["plan"]["selection"], json!([tower]));
            assert!(reply["jobs"].as_array().unwrap().is_empty());
        }
        assert!(selection(&registry, "embeddinggemma-2", "video").is_err());
        assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    }
    #[test]
    fn qwen_image_mlx_native_plan_contains_the_complete_pipeline_without_gguf_companions() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::new(dir.path().into()).with_backend("metal");
        assert_eq!(
            selection(&registry, "qwen-image-2.1", "mlx-4bit").unwrap(),
            ["mlx-4bit"]
        );
        let result: Value = serde_json::from_str(
            &execute(
                &registry,
                Command::Plan {
                    model: "qwen-image-2.1".into(),
                    artifact: "mlx-4bit".into(),
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(result["plan"]["file_count"], 23);
        assert_eq!(result["plan"]["selection"], json!(["mlx-4bit"]));
        assert!(result["jobs"].as_array().unwrap().is_empty());
    }
    #[test]
    fn qwen_mlx_download_includes_default_dflash2_without_vision_or_mtp() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::new(dir.path().into()).with_backend("metal");
        assert_eq!(
            selection(&registry, "qwen3.8-27b", "mlx-4bit").unwrap(),
            ["mlx-4bit", "drafter2"]
        );
        let result: Value = serde_json::from_str(
            &execute(
                &registry,
                Command::Plan {
                    model: "qwen3.8-27b".into(),
                    artifact: "mlx-4bit".into(),
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(result["plan"]["file_count"], 17);
        assert_eq!(result["plan"]["selection"], json!(["mlx-4bit", "drafter2"]));
        assert!(result["jobs"].as_array().unwrap().is_empty());
    }
    #[test]
    fn every_speech_model_has_a_native_plan_without_downloading() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::new(dir.path().into()).with_backend("metal");
        let mut models = Vec::new();
        for model in &registry.catalog().models {
            for weights in model.weights().filter(|a| {
                a.runtime.supports_backend("metal")
                    && a.capabilities(model).iter().any(|c| c == "transcription")
            }) {
                let response: Value = serde_json::from_str(
                    &execute(
                        &registry,
                        Command::Plan {
                            model: model.id.clone(),
                            artifact: weights.id.clone(),
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
                assert_eq!(response["plan"]["selection"][0], weights.id);
                assert!(response["plan"]["total"].as_u64().unwrap() > 0);
                assert!(response["jobs"].as_array().unwrap().is_empty());
                models.push(model.id.as_str());
            }
        }
        assert_eq!(models.len(), 6);
        assert!(models.contains(&"qwen3-asr-1.7b"));
        assert!(models.contains(&"kb-whisper-large"));
        assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    }

    #[test]
    fn native_selection_is_exact_and_backend_scoped() {
        let registry = Registry::new("/nonexistent-download-fixture".into()).with_backend("metal");
        let mut checked = 0;
        for model in &registry.catalog().models {
            for weights in model
                .weights()
                .filter(|a| a.runtime.supports_backend("metal"))
            {
                let ids = selection(&registry, &model.id, &weights.id).unwrap();
                assert_eq!(ids[0], weights.id);
                for id in &ids[1..] {
                    let a = model.artifact(id).unwrap();
                    assert!(
                        a.runtime.supports_backend("metal") && weights.runtime.allows_companion(id)
                    );
                    assert!(a.default || a.required);
                }
                checked += 1;
            }
        }
        assert!(checked > 5);
        for invalid in [
            r#"{"kind":"pull","model":"x","artifact":"y","selection":[],"url":"https://example.com"}"#,
            r#"{"kind":"list","key":"secret"}"#,
            r#"{"kind":"delete","id":"x"}"#,
            r#"{"kind":"delete","model":"x","artifact":"y","path":"/tmp"}"#,
        ] {
            assert!(serde_json::from_str::<Command>(invalid).is_err());
        }
        assert!(
            serde_json::from_str::<Command>(
                r#"{"kind":"delete","model":"qwen3.8-27b","artifact":"mlx-4bit"}"#
            )
            .is_ok()
        );
    }

    fn tiny_delete_registry(dir: &std::path::Path) -> Registry {
        let mut model = Registry::new(dir.into())
            .with_backend("metal")
            .catalog_of("embeddinggemma-2")
            .unwrap()
            .clone();
        model.id = "fixture".into();
        let mut weights = model.artifact("q8").unwrap().clone();
        weights.id = "q8".into();
        weights.files = vec![paddock_manager::registry::CatalogFile {
            dest: "fixture/q8.gguf".into(),
            url: "https://example.invalid/q8.gguf".into(),
            size: 4,
            sha256: "1".repeat(64),
        }];
        let mut alt = weights.clone();
        alt.id = "q8-alt".into();
        alt.default = false;
        alt.files = vec![paddock_manager::registry::CatalogFile {
            dest: "fixture/q8-alt.gguf".into(),
            url: "https://example.invalid/q8-alt.gguf".into(),
            size: 4,
            sha256: "2".repeat(64),
        }];
        let mut vision = model.artifact("vision").unwrap().clone();
        vision.files = vec![paddock_manager::registry::CatalogFile {
            dest: "fixture/vision.gguf".into(),
            url: "https://example.invalid/vision.gguf".into(),
            size: 4,
            sha256: "3".repeat(64),
        }];
        model.artifacts = vec![weights, alt, vision];
        Registry::from_catalog(
            paddock_manager::registry::Catalog {
                schema: 3,
                models: vec![model],
            },
            dir.into(),
        )
        .with_backend("metal")
    }

    fn install(registry: &Registry, ids: &[&str]) {
        let model = registry.catalog_of("fixture").unwrap();
        for id in ids {
            for file in &model.artifact(id).unwrap().files {
                let path = registry.models_dir().join(&file.dest);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, vec![7u8; file.size as usize]).unwrap();
            }
        }
    }

    #[test]
    fn native_delete_removes_weights_and_unshared_companions() {
        let dir = tempfile::tempdir().unwrap();
        let registry = tiny_delete_registry(dir.path());
        install(&registry, &["q8", "vision"]);
        let reply: Value = serde_json::from_str(
            &execute(
                &registry,
                Command::Delete {
                    model: "fixture".into(),
                    artifact: "q8".into(),
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(reply["deleted"]["model"], "fixture");
        assert_eq!(reply["deleted"]["artifact"], "q8");
        assert_eq!(reply["deleted"]["removed_files"], 2);
        assert_eq!(reply["deleted"]["kept_shared_files"], 0);
        let model = registry.catalog_of("fixture").unwrap();
        assert!(!registry.is_artifact_installed(model.artifact("q8").unwrap()));
        assert!(!registry.is_artifact_installed(model.artifact("vision").unwrap()));
        assert!(!dir.path().join("fixture").exists());
    }

    #[test]
    fn native_delete_keeps_companion_needed_by_another_installed_export() {
        let dir = tempfile::tempdir().unwrap();
        let registry = tiny_delete_registry(dir.path());
        install(&registry, &["q8", "q8-alt", "vision"]);
        execute(
            &registry,
            Command::Delete {
                model: "fixture".into(),
                artifact: "q8".into(),
            },
        )
        .unwrap();
        let model = registry.catalog_of("fixture").unwrap();
        assert!(!registry.is_artifact_installed(model.artifact("q8").unwrap()));
        assert!(registry.is_artifact_installed(model.artifact("q8-alt").unwrap()));
        assert!(registry.is_artifact_installed(model.artifact("vision").unwrap()));
    }
}
