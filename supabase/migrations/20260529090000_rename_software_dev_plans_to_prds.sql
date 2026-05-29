-- Rename the table and clean up columns that no longer apply to the PRD model.
alter table public.software_dev_plans rename to prds;

-- Drop stale columns that were specific to the old software-dev-plan pipeline.
alter table public.prds
    drop column if exists repo_analysis_json,
    drop column if exists planning_options_json;

-- Rename the index to match the new table name.
alter index if exists idx_software_dev_plans_owner_workspace_created
    rename to idx_prds_owner_workspace_created;
