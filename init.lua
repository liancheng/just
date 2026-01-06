vim.lsp.config["joule"] = {
	cmd = { "joule", "serve" },
	filetypes = { "jsonnet" },
	root_markers = { "vendor", "jsonnetfile.json", ".git" },
}

vim.lsp.enable("joule", true)
vim.lsp.inlay_hint.enable(true)
vim.lsp.set_log_level("TRACE")
