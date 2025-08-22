use comfy_table::{presets::ASCII_MARKDOWN, Cell, ContentArrangement, Table};
use breeze::app::{Project, Task};

fn truncate(s: &str, max: usize) -> String {
  if s.len() <= max { s.to_string() } else { format!("{}…", &s[..max.saturating_sub(1)]) }
}

pub enum RenderMode<'a> {
  Table,
  Tsv,
  Block,
  Json(&'a str), // not used here; JSON handled in main
}

pub fn projects_render(projects: &[Project], columns: &[String], mode: RenderMode<'_>) -> String {
  let normalized: Vec<String> = if columns.is_empty() {
    vec!["id","name","directory","description"].into_iter().map(|s| s.to_string()).collect()
  } else {
    columns.iter().map(|s| s.to_lowercase()).collect()
  };

  match mode {
    RenderMode::Tsv => {
      let mut out = String::new();
      out.push_str(&normalized.join("\t"));
      out.push('\n');
      for p in projects {
        let mut row: Vec<String> = Vec::new();
        for c in &normalized {
          let v = match c.as_str() {
            "id" => p.id.to_string(),
            "name" => p.name.clone(),
            "directory" => p.directory.clone(),
            "description" => p.description.clone().unwrap_or_default(),
            _ => String::new(),
          };
          row.push(v);
        }
        out.push_str(&row.join("\t"));
        out.push('\n');
      }
      out
    }
    RenderMode::Block => {
      // Legacy-style block output
      let mut out = String::new();
      if projects.is_empty() {
        return "No projects found.".to_string();
      }
      out.push_str("Projects:\n");
      for p in projects {
        out.push_str(&format!("\n  ID: {}\n  Name: {}\n  Directory: {}\n", p.id, p.name, p.directory));
        if let Some(desc) = &p.description {
          out.push_str(&format!("  Description: {}\n", desc));
        }
      }
      out
    }
    RenderMode::Table => {
      let mut table = Table::new();
      table
        .load_preset(ASCII_MARKDOWN)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(normalized.iter().map(|h| Cell::new(h.to_uppercase())).collect::<Vec<_>>());

      for p in projects {
        let mut row: Vec<Cell> = Vec::new();
        for c in &normalized {
          let v = match c.as_str() {
            "id" => p.id.to_string(),
            "name" => truncate(&p.name, 40),
            "directory" => truncate(&p.directory, 60),
            "description" => truncate(&p.description.clone().unwrap_or_default(), 60),
            _ => String::new(),
          };
          row.push(Cell::new(v));
        }
        table.add_row(row);
      }

      table.to_string()
    }
    RenderMode::Json(_) => unreachable!(),
  }
}

pub fn tasks_render(tasks: &[Task], columns: &[String], mode: RenderMode<'_>) -> String {
  let normalized: Vec<String> = if columns.is_empty() {
    vec!["id","project","status","created","files"].into_iter().map(|s| s.to_string()).collect()
  } else {
    columns.iter().map(|s| s.to_lowercase()).collect()
  };

  match mode {
    RenderMode::Tsv => {
      let mut out = String::new();
      out.push_str(&normalized.join("\t"));
      out.push('\n');
      for t in tasks {
        let mut row: Vec<String> = Vec::new();
        for c in &normalized {
          let v = match c.as_str() {
            "id" => t.id.to_string(),
            "project" => t.project_id.to_string(),
            "status" => t.status.to_string(),
            "created" => t.created_at.to_string(),
            "files" => t.files_indexed.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string()),
            _ => String::new(),
          };
          row.push(v);
        }
        out.push_str(&row.join("\t"));
        out.push('\n');
      }
      out
    }
    RenderMode::Block => {
      let mut out = String::new();
      if tasks.is_empty() {
        return "No tasks found.".to_string();
      }
      out.push_str("Tasks:\n");
      for t in tasks {
        out.push_str(&format!(
          "\n  ID: {}\n  Project: {}\n  Status: {}\n  Created: {}\n",
          t.id, t.project_id, t.status, t.created_at
        ));
        if let Some(files) = t.files_indexed {
          out.push_str(&format!("  Files indexed: {}\n", files));
        }
      }
      out
    }
    RenderMode::Table => {
      let mut table = Table::new();
      table
        .load_preset(ASCII_MARKDOWN)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(normalized.iter().map(|h| Cell::new(h.to_uppercase())).collect::<Vec<_>>());

      for t in tasks {
        let mut row: Vec<Cell> = Vec::new();
        for c in &normalized {
          let v = match c.as_str() {
            "id" => t.id.to_string(),
            "project" => t.project_id.to_string(),
            "status" => t.status.to_string(),
            "created" => t.created_at.to_string(),
            "files" => t.files_indexed.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string()),
            _ => String::new(),
          };
          row.push(Cell::new(v));
        }
        table.add_row(row);
      }

      table.to_string()
    }
    RenderMode::Json(_) => unreachable!(),
  }
}
