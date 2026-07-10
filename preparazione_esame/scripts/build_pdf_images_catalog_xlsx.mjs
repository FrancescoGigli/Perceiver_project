import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const OUT_DIR = "C:\\Users\\gigli\\Desktop\\PDF_Bibliografia_Perceiver_immagini_estratte";
const INDEX_CSV = path.join(OUT_DIR, "tabella_immagini_pdf.csv");
const SUMMARY_CSV = path.join(OUT_DIR, "riepilogo_pdf.csv");
const XLSX_OUT = path.join(OUT_DIR, "tabella_immagini_pdf.xlsx");

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  const [headers, ...data] = rows;
  return data
    .filter((r) => r.some((v) => v !== ""))
    .map((r) => Object.fromEntries(headers.map((h, idx) => [h, r[idx] ?? ""])));
}

function n(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function colName(index) {
  let name = "";
  let nIndex = index + 1;
  while (nIndex > 0) {
    const rem = (nIndex - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    nIndex = Math.floor((nIndex - 1) / 26);
  }
  return name;
}

function rangeAddress(startRow, startCol, rowCount, colCount) {
  const a = `${colName(startCol)}${startRow + 1}`;
  const b = `${colName(startCol + colCount - 1)}${startRow + rowCount}`;
  return `${a}:${b}`;
}

const indexRows = parseCsv(await fs.readFile(INDEX_CSV, "utf8"));
const summaryRows = parseCsv(await fs.readFile(SUMMARY_CSV, "utf8"));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Riepilogo");
const index = workbook.worksheets.add("Indice immagini");
summary.showGridLines = false;
index.showGridLines = false;

const totalImages = indexRows.length;
const totalPdfs = summaryRows.length;
const totalMasks = summaryRows.reduce((acc, r) => acc + n(r["Maschere escluse"]), 0);
const pdfsWithoutImages = summaryRows.filter((r) => n(r["Immagini estratte"]) === 0).length;
const imagesWithNotes = indexRows.filter((r) => r.Note).length;

summary.getRange("A1:E1").merge();
summary.getRange("A1").values = [["Catalogo immagini estratte dai PDF"]];
summary.getRange("A1").format = {
  fill: "#17324D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
};
summary.getRange("A3:B8").values = [
  ["Cartella immagini", path.join(OUT_DIR, "immagini")],
  ["PDF processati", totalPdfs],
  ["Immagini raster estratte", totalImages],
  ["Maschere alfa escluse", totalMasks],
  ["PDF senza immagini raster", pdfsWithoutImages],
  ["Righe con nota frammento/icona", imagesWithNotes],
];
summary.getRange("B3:E3").merge();
summary.getRange("A3:A8").format = { font: { bold: true }, fill: "#EAF2F8" };
summary.getRange("B3:B8").format = { wrapText: true };

summary.getRange("A10:E10").values = [["PDF", "Immagini estratte", "Immagini trovate", "Maschere escluse", "Note"]];
summary.getRange("A10:E10").format = {
  fill: "#2F6F73",
  font: { bold: true, color: "#FFFFFF" },
};
const summaryMatrix = summaryRows.map((r) => [
  r.PDF,
  n(r["Immagini estratte"]),
  n(r["Immagini principali trovate"]),
  n(r["Maschere escluse"]),
  r.Note,
]);
if (summaryMatrix.length) {
  summary.getRangeByIndexes(10, 0, summaryMatrix.length, 5).values = summaryMatrix;
  summary.tables.add(`A10:E${10 + summaryMatrix.length}`, true, "RiepilogoPDF");
}
summary.getRange("A:A").format.columnWidthPx = 520;
summary.getRange("B:D").format.columnWidthPx = 120;
summary.getRange("E:E").format.columnWidthPx = 320;
summary.getRange(`A10:E${10 + Math.max(summaryMatrix.length, 1)}`).format.borders = {
  preset: "all",
  style: "thin",
  color: "#D7DEE8",
};
summary.freezePanes.freezeRows(10);

const headers = [
  "ID",
  "PDF",
  "Pagina",
  "N. immagine",
  "File immagine",
  "Descrizione",
  "Note",
  "Larghezza px",
  "Altezza px",
  "Encoding",
  "Object ID",
  "SHA1",
];
index.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
index.getRangeByIndexes(0, 0, 1, headers.length).format = {
  fill: "#17324D",
  font: { bold: true, color: "#FFFFFF" },
};

const indexMatrix = indexRows.map((r) => [
  n(r.ID),
  r.PDF,
  n(r.Pagina),
  n(r.Numero_immagine_pdf),
  r.File_immagine,
  r.Descrizione,
  r.Note,
  n(r.Larghezza_px),
  n(r.Altezza_px),
  r.Encoding,
  r.Object_ID,
  r.SHA1,
]);
if (indexMatrix.length) {
  index.getRangeByIndexes(1, 0, indexMatrix.length, headers.length).values = indexMatrix;
  index.tables.add(`A1:L${indexMatrix.length + 1}`, true, "IndiceImmagini");
}

const widthPx = [60, 340, 70, 90, 520, 720, 220, 95, 95, 90, 95, 300];
for (let c = 0; c < widthPx.length; c++) {
  const addr = rangeAddress(0, c, Math.max(indexMatrix.length + 1, 2), 1);
  index.getRange(addr).format.columnWidthPx = widthPx[c];
}
index.getRange(`A1:L${indexMatrix.length + 1}`).format.borders = {
  preset: "all",
  style: "thin",
  color: "#E0E6EF",
};
index.getRange(`B2:B${indexMatrix.length + 1}`).format = { wrapText: true };
index.getRange(`F2:G${indexMatrix.length + 1}`).format = { wrapText: true };
index.getRange(`A2:A${indexMatrix.length + 1}`).format.numberFormat = "0";
index.getRange(`C2:D${indexMatrix.length + 1}`).format.numberFormat = "0";
index.getRange(`H2:I${indexMatrix.length + 1}`).format.numberFormat = "0";
index.freezePanes.freezeRows(1);

await fs.mkdir(OUT_DIR, { recursive: true });
const summaryPreview = await workbook.render({ sheetName: "Riepilogo", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(
  path.join(OUT_DIR, "preview_riepilogo.png"),
  new Uint8Array(await summaryPreview.arrayBuffer()),
);
const indexPreview = await workbook.render({
  sheetName: "Indice immagini",
  range: "A1:L25",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(OUT_DIR, "preview_indice_immagini.png"),
  new Uint8Array(await indexPreview.arrayBuffer()),
);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(XLSX_OUT);
console.log(`Saved ${XLSX_OUT}`);
